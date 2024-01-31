
```
python analyser.py -h
usage: analyser.py [-h] --targets_file TARGETS_FILE --fastq_file FASTQ_FILE --outdir OUTDIR
                   [--prefix PREFIX] [--bc_start BC_START] [--bc_end BC_END]
                   [--bc_pattern BC_PATTERN] [--score_match SCORE_MATCH]
                   [--score_mismatch SCORE_MISMATCH] [--score_gap_open SCORE_GAP_OPEN]
                   [--score_gap_extend SCORE_GAP_EXTEND]

optional arguments:
  -h, --help            show this help message and exit
  --targets_file TARGETS_FILE
                        csv file which must contain "barcode", "target_seq", "edit_region_pos1",
                        "edit_region_pos2" four columns. edit_region_pos1/pos2: start/end position
                        of expected editting region in target sequence.
  --fastq_file FASTQ_FILE
                        merged fastq file or single-end fastq file
  --outdir OUTDIR       directory for output files
  --prefix PREFIX       prefix for output files
  --bc_start BC_START   barcode start position (0-base) in sequencing reads
  --bc_end BC_END       barcode end position (0-base) in sequencing reads (contained)
  --bc_pattern BC_PATTERN
                        regular expression of barcode
  --score_match SCORE_MATCH
                        match score for pairwise alignment between sequencing reads and target
                        sequence, default: 2
  --score_mismatch SCORE_MISMATCH
                        mismatch score for pairwise alignment between sequencing reads and target
                        sequence, default: -1
  --score_gap_open SCORE_GAP_OPEN
                        gap open score for pairwise alignment between sequencing reads and target
                        sequence, default: -2
  --score_gap_extend SCORE_GAP_EXTEND
                        gap extend score for pairwise alignment between sequencing reads and
                        target sequence, default: -1

```

example:
```
python analyser.py \
--targets_file test\test_targets_lib_info.csv 
--fastq_file test\test.fastq \
--bc_pattern "TTTG[ATCG]{15}" \
--bc_start 50 \
--bc_end 80 \
--outdir test\output \
--prefix test
```