# Dataset folder
This is the default folder used to read datasets by training and test scripts provided in this repository.

The complete set of datasets can be downloaded [here]( http://hdl.handle.net/2047/D20621419).

## Table of provided datasets

Overview of all datasets (DS) collected in this work. DS 1.0 and DS 1.1 consist of just baseband waveforms (random synthetic channel is applied dynamically at runtime). For OTA and wired collected signals, we use Ettus USRP X310 as the transmitter hardware and [NI-NEU RF Data Recording API (NI-API)](https://github.com/genesys-neu/ni-rf-data-recording-api) to control the USRP.

| Dataset ID     | Medium     | Tx / Rx          | Mode          | # of examples                         | Description                                           | Zip file    |
|--------|------------|------------------|---------------|---------------------------------------|-------------------------------------------------------|-----------------|
| 1.0 | Synthetic  | -                | Single protocol | 2000 for each protocol                | Single packet per file (unbalanced dataset)           | [DATASET1_0.zip](http://hdl.handle.net/2047/D20621420) |
| 1.1 | Synthetic  | -                | Single protocol | 2000 for each protocol                | Multiple packet per file (balanced dataset)           | [DATASET1_1.zip]( http://hdl.handle.net/2047/D20621421) |
| 2.0 | Wire       | NI-API           | Multi-protocol | 800 non-overlapping, 3300 overlapping | Wideband spectrogram. Overlapping and non-overlapping | [DATASET2_0.zip](http://hdl.handle.net/2047/D20621422) | 
| 3.0 | OTA        | NI-API / AIR-T   | Single protocol | 7279 transmission  | Signals collected in multiple rooms                   | [DATASET3_0.zip](http://hdl.handle.net/2047/D20621423) |
| 3.1 | OTA        | NI-API / AIR-T   | Multi-protocol | 400 for each overlap configuration    | 6 combinations. 25% and 50% overlap (bw=62.5 MHz)   | [DATASET3_1.zip]( http://hdl.handle.net/2047/D20621424) |
| 3.2 | OTA        | NI-API / AIR-T   | Multi-protocol | 200 for each overlap configuration    | 6 combinations. 25% and 50% overlap (bw=20 MHz)     | [DATASET3_2.zip](http://hdl.handle.net/2047/D20621425) |
| 3.3 | OTA        | NI-API / AIR-T   | Multi-protocol | 200 for each overlap configuration    | 12 combinations; 25% and 50% overlap (both bw=20 MHz & bw=62.5 MHz are provided)                 | [DATASET3_3.zip](http://hdl.handle.net/2047/D20621426) |
| 3.4 | OTA        | NI-API / AIR-T   | Single protocol | 700 for each protocol                 | Additional OTA collection (excluded from results)     | [DATASET3_4_1.zip](http://hdl.handle.net/2047/D20621427) [DATASET3_4_2.zip](http://hdl.handle.net/2047/D20621428) |


Once the desired dataset .zip file has been downloade, the following command can be used to unzip the dataset folders into `data/` folder:
```
unzip <dataset filename>.zip -d data/
```
The resulting structure of `data/` will look like this:
```
data/
├── ...
├── DATASET1_1
├── DATASET1_1_TEST
├── ...
└── README.md
```
