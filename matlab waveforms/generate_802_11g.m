

%This is the code for dataset 1_0 (due 01/18 for DSTL). No noise, channel
%model, just vanilla signal. 2k samples for each. Run from
%/home/genesys/Downloads/Stratis_Kaushik_Proposal/dataset1_0' directory 

trainDir = fullfile('/media/genesys/genesys/DSTL_DATASET_1_0/802_11g');
frameNo = 1;
row_count = 1;

% Generating 802.11n waveform
% 802.11n configuration

%% Generating 802.11a/g/j (OFDM) waveform
% 802.11a/g/j (OFDM) configuration
for frameNo = 1:1000 % first 1000 is 1/2 rate , 2nd 1000 is 3/4 rate

    nonHTCfg = wlanNonHTConfig('Modulation', 'OFDM', ...
        'ChannelBandwidth', 'CBW20', ...
        'MCS', 3, ...
        'PSDULength', 1000);
    
    numPackets = 1;
    % input bit source:
    in = randi([0, 1], 1000, 1);
    
    
    % Generation
    waveform = wlanWaveformGenerator(in, nonHTCfg, ...
        'NumPackets', numPackets, ...
        'IdleTime', 0, ...
        'OversamplingFactor', 1, ...
        'ScramblerInitialization', 93, ...
        'WindowTransitionTime', 1e-07);
    
    Fs = wlanSampleRate(nonHTCfg, 'OversamplingFactor', 1); 								 % Specify the sample rate of the waveform in Hz
    
    %% Visualize
%     Spectrum Analyzer
    spectrum = dsp.SpectrumAnalyzer('SampleRate', Fs);
    spectrum(waveform);
    release(spectrum);

    % Save IQ
    label = '802.11g_IQ';
    fname3 = fullfile(trainDir, [label '_frame_' strrep(num2str(frameNo),' ','')]);
    % write_binary (fname, 2, waveform)
    save(fname3 + ".mat", 'waveform')
    
    fileID = fopen(fname3 + ".yaml",'w');
         metadata = strcat('# ---------------------------------------------------------------------,\n', ...
            '# Waveform Configuration]\n', ...
            '# ---------------------------------------------------------------------,]\n', ...
            'standard: "802.11g_matlab"\n', ...
            'modulation: "OFDM"\n', ... 
            'MCS: "16-QAM(1/2 rate)"\n', ... %add a flag to auto change for 1/2 vs 3/4 rate, automate this
            'bandwidth: ' + " " ,num2str(Fs), '\n', ... 
            'sampling_rate: ' + " " ,num2str(Fs));
%             'PSDU_length_bytes: 1024');  
         %             'time_duration: ' + " ", num2str(frameDuration), '\n', ... %is this needed ?

        %add other MIMO specific things? maybe create another category for the
        %MIMO related WIFI in wireless_link_paramter.yaml    
        fprintf(fileID, metadata);
        fclose(fileID);
    
end 

%for MCS 3/4 RATE 
for frameNo = 1001:2000 % first 1000 is 1/2 rate , 2nd 1000 is 3/4 rate

    nonHTCfg = wlanNonHTConfig('Modulation', 'OFDM', ...
        'ChannelBandwidth', 'CBW20', ...
        'MCS', 4, ...
        'PSDULength', 1000);
    
    numPackets = 1;
    % input bit source:
    in = randi([0, 1], 1000, 1);
    
    
    % Generation
    waveform = wlanWaveformGenerator(in, nonHTCfg, ...
        'NumPackets', numPackets, ...
        'IdleTime', 0, ...
        'OversamplingFactor', 1, ...
        'ScramblerInitialization', 93, ...
        'WindowTransitionTime', 1e-07);
    
    Fs = wlanSampleRate(nonHTCfg, 'OversamplingFactor', 1); 								 % Specify the sample rate of the waveform in Hz
    
    %% Visualize
    % Spectrum Analyzer
%     spectrum = dsp.SpectrumAnalyzer('SampleRate', Fs);
%     spectrum(waveform);
%     release(spectrum);

    % Save IQ
    % Save IQ
    label = '802.11g_IQ';
    fname3 = fullfile(trainDir, [label '_frame_' strrep(num2str(frameNo),' ','')]);
    % write_binary (fname, 2, waveform)
    save(fname3 + ".mat", 'waveform')
    
    fileID = fopen(fname3 + ".yaml",'w');
         metadata = strcat('# ---------------------------------------------------------------------,\n', ...
            '# Waveform Configuration]\n', ...
            '# ---------------------------------------------------------------------,]\n', ...
            'standard: "802.11g_matlab"\n', ...
            'modulation: "OFDM"\n', ... 
            'MCS: "16-QAM(3/4 rate)"\n', ... %add a flag to auto change for 1/2 vs 3/4 rate, automate this
            'bandwidth: ' + " " ,num2str(Fs), '\n', ... 
            'sampling_rate: ' + " " ,num2str(Fs));
%             'PSDU_length_bytes: 1024');  
         %             'time_duration: ' + " ", num2str(frameDuration), '\n', ... %is this needed ?

        %add other MIMO specific things? maybe create another category for the
        %MIMO related WIFI in wireless_link_paramter.yaml    
        fprintf(fileID, metadata);
        fclose(fileID);
    
end 

