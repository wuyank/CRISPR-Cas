import os
import gzip
import regex
import pandas as pd
from Bio import pairwise2
import argparse
import time

def localtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 

def open_fq(file):
    if file.endswith('.gz'):
        fq = gzip.open(file,'rb')
        if_decode = 1
    else:
        fq = open(file,'r')
        if_decode = 0
    with fq as f:
        while True:
            l1 = f.readline().strip()
            if not l1:
                break
            l2 = f.readline().strip()
            l3 = f.readline().strip()
            l4 = f.readline().strip()
            if if_decode:
                l1 = l1.decode('utf-8').strip()
                l2 = l2.decode('utf-8').strip()
                l3 = l3.decode('utf-8').strip()
                l4 = l4.decode('utf-8').strip()
            yield [l1,l2,l3,l4]

def demultiplex_fq_bc_range(fqfile, bc_start, bc_end):
    # bc_start, bc_end: 0-base, 包含bc_end
    demultiplex_dict = {}
    for l1,l2,l3,l4 in open_fq(fqfile):
        bc = l2[bc_start : bc_end + 1]
        bc_dict = demultiplex_dict.get(bc, {})
        bc_dict.setdefault('count',0)
        bc_dict['count'] += 1
        bc_dict.setdefault('seqs',[])
        bc_dict['seqs'].append(l2)
        demultiplex_dict[bc] = bc_dict
    return demultiplex_dict

def demultiplex_fq_bc_pattern(fqfile,bc_pattern):
    demultiplex_dict = {}
    for l1,l2,l3,l4 in open_fq(fqfile):
        matches = regex.finditer(bc_pattern, l2, overlapped=True)
        for m in matches:
            bc = m.group()
            bc_dict = demultiplex_dict.get(bc, {})
            bc_dict.setdefault('count',0)
            bc_dict['count'] += 1
            bc_dict.setdefault('seqs',[])
            bc_dict['seqs'].append(l2)
            demultiplex_dict[bc] = bc_dict
    return demultiplex_dict

def demultiplex_fq_bc_range_pattern(fqfile, bc_start, bc_end, bc_pattern):
    # bc_start, bc_end: 0-base, 包含bc_end
    demultiplex_dict = {}
    for l1,l2,l3,l4 in open_fq(fqfile):
        bc_region = l2[bc_start : bc_end + 1]
        matches = regex.finditer(bc_pattern, bc_region, overlapped=True)
        for m in matches:
            bc = m.group()
            bc_dict = demultiplex_dict.get(bc, {})
            bc_dict.setdefault('count',0)
            bc_dict['count'] += 1
            bc_dict.setdefault('seqs',[])
            bc_dict['seqs'].append(l2)
            demultiplex_dict[bc] = bc_dict
    return demultiplex_dict

def align(seq, target, score_match=2, score_mismatch=-1, score_gap_open=-2, score_gap_extend=-1):
    al = pairwise2.align.globalms(seq, target, score_match, score_mismatch, score_gap_open, score_gap_extend)
    aligned_seq, aligned_target = al[0][:2]
    target_pos = 0
    substitution = []
    insertion = []
    deletion = []
    for s,t in zip(aligned_seq, aligned_target):
        if t != '-':
            target_pos += 1
        if s != t:
            if s != '-' and t != '-':
                substitution.append(f'pos{target_pos}_{t}sub{s}')
            elif s == '-':
                deletion.append(f'pos{target_pos}_{t}del')
            elif t == '-':
                insertion.append(f'pos{target_pos}_ins{s}')

    return aligned_seq, aligned_target, substitution, insertion, deletion
        
def check_editing(target_quantification_pos, sub, ins, dele):
    insertion = ins.split(',')
    deletion = dele.split(',')
    substitution = sub.split(',')
    indel_list = insertion + deletion
    subsindel_list = substitution + indel_list
    indel_pos = set([int(x.split('_')[0][3:]) for x in indel_list if x])
    subsindel_pos = set([int(x.split('_')[0][3:]) for x in subsindel_list if x])
    quantification_pos = set(range(target_quantification_pos[0],target_quantification_pos[1]+1))

    # 筛选在quantification_pos范围内只有一个indel的read
    if len(indel_pos & quantification_pos) == 1:
        filter = 'T'
    else:
        filter = 'F'

    if indel_pos & quantification_pos:
        indel_editing = 'T'
    else:
        indel_editing = 'F'
    if subsindel_pos & quantification_pos:
        subsindel_editing = 'T'
    else:
        subsindel_editing = 'F'
    return (indel_editing,subsindel_editing,filter)

class Analyser():
    def __init__(self, 
                 targets_file, 
                 fastq_file, 
                 outdir,
                 prefix,
                 bc_start=None, 
                 bc_end=None,
                 bc_pattern=None,
                 score_match=2, 
                 score_mismatch=-1, 
                 score_gap_open=-2, 
                 score_gap_extend=-1
                 ) -> None:
        self.targets_df = pd.read_csv(targets_file)
        self.fastq = fastq_file
        self.edit_region = {bc:(pos1,pos2) for bc,pos1,pos2 in zip(self.targets_df['barcode'], self.targets_df['edit_region_pos1'], self.targets_df['edit_region_pos2'])}
        self.outdir = outdir
        self.check_dir(self.outdir)
        if prefix:
            self.aligned_out = f'{self.outdir}/{prefix}.alignment.txt'
            self.indel_freq_out = f'{self.outdir}/{prefix}.indel_freq.csv'
        else:
            self.aligned_out = f'{self.outdir}/alignment.txt'
            self.indel_freq_out = f'{self.outdir}/indel_freq.csv'
        if (bc_start is None) ^ (bc_end is None):
            raise ValueError('Only one barcode position was specified, you must specified "bc_start" and "bc_end" together or using "bc_pattern" to specify barcode.')
        else:
            if bc_start is None:
                self.using_bc_range = False
            else:
                self.using_bc_range = True
                self.bc_start = bc_start
                self.bc_end = bc_end
        if bc_pattern:
            self.using_bc_pattern = True
            self.bc_pattern = bc_pattern
        else:
            self.using_bc_pattern = False
        if not (self.using_bc_range or self.using_bc_pattern):
            raise ValueError('Please specify barcode range by "bc_start" and "bc_end" or specify barcode pattern by "bc_pattern"')

        self.score_match = score_match
        self.score_mismatch = score_mismatch
        self.score_gap_open = score_gap_open
        self.score_gap_extend = score_gap_extend
    
    def check_dir(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)


    def demultiplex(self):
        if self.using_bc_range and (not self.using_bc_pattern):
            print(localtime(), 'Demultiplex fastq using barcode range...')
            self.demultiplexed_fq = demultiplex_fq_bc_range(self.fastq, self.bc_start, self.bc_end)
        elif self.using_bc_pattern and (not self.using_bc_range):
            print(localtime(), 'Demultiplex fastq using barcode pattern...')
            self.demultiplexed_fq = demultiplex_fq_bc_pattern(self.fastq, self.bc_pattern)
        else:
            print(localtime(), 'Demultiplex fastq using barcode range and pattern together...')
            self.demultiplexed_fq = demultiplex_fq_bc_range_pattern(self.fastq, self.bc_start, self.bc_end, self.bc_pattern)
        print(localtime(), 'Demultiplex fastq done.')
    
    def align(self):
        print(localtime(), 'Start align reads to target sequences...')
        allreads = 0
        for bc, bc_dict in self.demultiplexed_fq.items():
            allreads += bc_dict['count']
        outf = open(self.aligned_out, 'w')
        header = 'barcode\treads_count\taligned_seq\taligned_target\tsubstitution\tinsertion\tdeletion\n'
        outf.write(header)
        num = 0
        for bc, target in zip(self.targets_df['barcode'], self.targets_df['target_seq']):
            reads_num = self.demultiplexed_fq.get(bc, {}).get('count',0)
            if reads_num != 0:
                reads = self.demultiplexed_fq[bc]['seqs']
                for seq in reads:
                    num += 1
                    aligned_seq, aligned_target, substitution, insertion, deletion = align(seq,
                                                                                           target,
                                                                                           self.score_match,
                                                                                           self.score_mismatch,
                                                                                           self.score_gap_open,
                                                                                           self.score_gap_extend)
                    line = f'{bc}\t{reads_num}\t{aligned_seq}\t{aligned_target}\t{",".join(substitution)}\t{",".join(insertion)}\t{",".join(deletion)}\n'
                    outf.write(line)
                    if num % 10000 == 0:
                        print(localtime(), f'processed reads num: {num} / {allreads}')
            else:
                line = f'{bc}\t{0}\t\t\t\t\t\n'
                outf.write(line)
        print(localtime(), 'Align done.')
        outf.close()

    def indel_statistic(self):
        print(localtime(), 'Start indel frequency statistic...')
        align_df = pd.read_csv(self.aligned_out, sep='\t')
        align_df.fillna('', inplace=True)
        read_count_dict = dict(zip(align_df['barcode'],align_df['reads_count']))
        check_list = []
        for bc, sub, ins, dele in zip(align_df['barcode'],align_df['substitution'],align_df['insertion'],align_df['deletion']):
            check_list.append(check_editing(self.edit_region[bc], sub, ins, dele))
        align_df['check_indel_editing'] = [x[0] for x in check_list]
        align_df['check_subsindel_editing'] = [x[1] for x in check_list]
        align_df['check_filter'] = [x[2] for x in check_list]
        indel_read_edit_dict = dict((align_df['barcode'] + '_' + align_df['check_indel_editing']).value_counts())
        subsindel_read_edit_dict = dict((align_df['barcode'] + '_' + align_df['check_subsindel_editing']).value_counts())
        filter_dict = dict((align_df['barcode'] + '_' + align_df['check_filter']).value_counts())
        self.targets_df['all_readcount'] = self.targets_df['barcode'].apply(lambda x: read_count_dict[x])
        self.targets_df[f'indel_readcount'] = self.targets_df['barcode'].apply(lambda x: indel_read_edit_dict.get(x+'_T',0))
        self.targets_df[f'indel_freq'] = self.targets_df[f'indel_readcount'] / self.targets_df[f'all_readcount']
        self.targets_df[f'subsindel_readcount'] = self.targets_df['barcode'].apply(lambda x: subsindel_read_edit_dict.get(x+'_T',0))
        self.targets_df[f'subsindel_freq'] = self.targets_df[f'subsindel_readcount'] / self.targets_df[f'all_readcount']
        self.targets_df[f'filter_readcount'] = self.targets_df['barcode'].apply(lambda x: filter_dict.get(x+'_T', 0))
        self.targets_df[f'indel_filtered_freq'] = (self.targets_df[f'indel_readcount'] - self.targets_df[f'filter_readcount']) / (self.targets_df[f'all_readcount'] - self.targets_df[f'filter_readcount'])
        self.targets_df[f'subsindel_filtered_freq'] = (self.targets_df[f'subsindel_readcount'] - self.targets_df[f'filter_readcount']) / (self.targets_df[f'all_readcount'] - self.targets_df[f'filter_readcount'])
        self.targets_df.to_csv(self.indel_freq_out, index=False)
        print(localtime(), 'Done.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets_file',
                        required=True,
                        help='csv file which must contain "barcode", "target_seq", "edit_region_pos1", "edit_region_pos2" four columns. edit_region_pos1/pos2: start/end position of expected editting region in target sequence.')
    parser.add_argument('--fastq_file',
                        required=True,
                        help='merged fastq file or single-end fastq file')
    parser.add_argument('--outdir',
                        required=True,
                        help='directory for output files')
    parser.add_argument('--prefix',
                        default='',
                        help='prefix for output files')
    parser.add_argument('--bc_start',
                        type=int,
                        default=None,
                        help='barcode start position (0-base) in sequencing reads')
    parser.add_argument('--bc_end',
                        type=int,
                        default=None,
                        help='barcode end position (0-base) in sequencing reads (contained)')
    parser.add_argument('--bc_pattern',
                        type=str,
                        default=None,
                        help='regular expression of barcode')
    parser.add_argument('--score_match',
                        type=int,
                        default=2,
                        help='match score for pairwise alignment between sequencing reads and target sequence, default: 2'
                        )
    parser.add_argument('--score_mismatch',
                        type=int,
                        default=-1,
                        help='mismatch score for pairwise alignment between sequencing reads and target sequence, default: -1'
                        )
    parser.add_argument('--score_gap_open',
                        type=int,
                        default=-2,
                        help='gap open score for pairwise alignment between sequencing reads and target sequence, default: -2')
    parser.add_argument('--score_gap_extend',
                        type=int,
                        default=-1,
                        help='gap extend score for pairwise alignment between sequencing reads and target sequence, default: -1')
    return parser.parse_args()



def main():
    args = parse_args()
    analyser = Analyser(
        targets_file = args.targets_file,
        fastq_file = args.fastq_file,
        outdir = args.outdir,
        prefix = args.prefix,
        bc_start = args.bc_start,
        bc_end = args.bc_end,
        bc_pattern = args.bc_pattern,
        score_match = args.score_match,
        score_mismatch = args.score_mismatch,
        score_gap_open = args.score_gap_open,
        score_gap_extend = args.score_gap_extend
    )
    analyser.demultiplex()
    analyser.align()
    analyser.indel_statistic()


if __name__ == '__main__':
    main()


