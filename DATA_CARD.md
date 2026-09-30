# Data Card

## Dataset Scope

The dataset is an internal experimental collection of short violin-performance
clips. Source videos were collected from publicly accessible violin performance
and teaching videos on Bilibili and YouTube, then processed locally to extract
WAV audio and MediaPipe skeleton CSV files. Each clip is assigned one dominant
technique label:

- pizzicato: plucked-string articulation
- stopped bowing / dungong: short separated bowed notes with clear stop-and-start attacks
- plain playing: ordinary bowed playing without the five target special techniques
- vibrato: periodic left-hand pitch modulation
- double stops: two strings sounded simultaneously
- spiccato: bounced off-string bowing

Tremolo is excluded from the reported six-class experiment because only one
usable sample is currently available.

## Annotation

Each clip is assigned one author-provided dominant technique label. Annotation
was guided by the source context, visible bowing or fingering motion, and
auditory characteristics. The labels have not yet been independently verified by
a professional violin teacher or a second annotator. A stronger future dataset
should include multiple annotators and report inter-annotator agreement.

As an additional internal quality check, a stratified subset of 48 clips was
audited with source-video-grouped out-of-fold model suggestions. The suggestions
matched the original labels for 45 of the 48 audited clips (93.75%), and three
clips were flagged for manual review. This is an internal consistency check, not
independent human annotation.

## Current Usable Distribution

| Class | Clips |
| --- | ---: |
| double_stop | 23 |
| dungong | 25 |
| pizzicato | 27 |
| plain | 33 |
| spiccato | 21 |
| vibrato | 18 |
| Total | 147 |

## Manifest

The experiment script writes `tmp_out/data_manifest.csv`, which contains:

- label
- original folder
- sample id
- sample name
- source video path
- source group
- prepared wav path
- skeleton CSV path
- skeleton CSV type

This manifest is intended to make the experiment auditable and reduce accidental
data leakage from stale or unmatched files.

## Use and Release Notes

The raw videos and prepared audio files are not intended for public GitHub
release. Public redistribution would require checking platform terms, copyright,
creator permission, consent, and privacy status. For a public repository, release
only code, aggregate results, model cards, paper figures, and a small anonymized
example if permission is available. A stronger future dataset should use
self-recorded clips or clips with explicit open licenses.

## Known Limitations

The current source-group split assigns all clips with the same recoverable
source-video path to the same fold. Because the cleaned dataset currently maps
147 usable clips to 147 recoverable source-video groups, this audit mainly
checks duplicate source-file leakage. It is not yet a verified
performer-disjoint or piece-disjoint split. Future data collection should add
performer id, piece id, recording source, camera setup, and independent
annotator information.
