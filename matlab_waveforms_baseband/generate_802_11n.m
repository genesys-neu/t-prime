%This is the code for dataset 1_0 (due 01/18 for DSTL). No noise, channel
%model, just vanilla signal. 2k samples for each 

%This is the code for dataset 1_0 (due 01/18 for DSTL). No noise, channel
%model, just vanilla signal. 2k samples for each. Run from
%/home/genesys/Downloads/Stratis_Kaushik_Proposal/dataset1_0' directory 

% trainDir = fullfile(pwd,'802_11n');
trainDir = fullfile('/media/genesys/Seagate Portable Drive/test_data_1_1/802_11n');
frameNo = 1;
row_count = 1;

% Generating 802.11n waveform
% 802.11n configuration
for frameNo = 1:250 % first 1000 is 1/2 rate , 2nd 1000 is 3/4 rate

   vhtCfg = wlanVHTConfig('ChannelBandwidth', 'CBW20', ...
    'NumUsers', 1, ...
    'NumTransmitAntennas', 1, ...
    'NumSpaceTimeStreams', [1], ...
    'SpatialMapping', 'Direct', ...
    'STBC', false, ...
    'MCS', 3, ...
    'ChannelCoding', 'BCC', ...
    'APEPLength', 1024, ...
    'GuardInterval', 'Short', ...
    'GroupID', 63, ...
    'PartialAID', 275);
    
    %Long
    numPackets = 5;
    % input bit source:
    in = randi([0, 1], 1000, 1);
    
    
    % Generation
    waveform = wlanWaveformGenerator(in, vhtCfg, ...
    'NumPackets', numPackets, ...
    'IdleTime', 0, ...
    'OversamplingFactor', 1, ...
    'ScramblerInitialization', 93, ...
    'WindowTransitionTime', 1e-07);

    Fs = wlanSampleRate(vhtCfg, 'OversamplingFactor', 1); 	
    
    
   
    
    %% Visualize
    % Spectrum Analyzer
%     spectrum = dsp.SpectrumAnalyzer('SampleRate', Fs, ...
%         'NumInputPorts',1, ...
%         'ShowLegend',true);
%     spectrum(waveform); %both
%     release(spectrum);
    
    %create spectrogram and PSD 
%     saveSpectrogramImage_Vini_RxScript(waveform,20e6,0,[20e6],'/home/genesys/Downloads/Stratis_Kaushik_Proposal/dataset1_0/802_11n/specs','802.11n_IQ',[256 256],1);   
%     rxWave,sr,fc,BW, folder,label,imageSize, idx (8)

    
    %% SAVE WAVEFORM
    
%     saveSpectrogramImage_Vini_RxScript(waveform,20e6,0,[20e6],'/home/genesys/Downloads/Stratis_Kaushik_Proposal/dataset1_0/802_11n/specs', '802.11n_IQ',[256 256],1);   
    %TO DO: REMOVE PSD PART IN PLOTS ??

    % % Metadata File
    % label = '802.11g_Metadata';
    % fname2 = fullfile(trainDir, [label '_frame_' strrep(num2str(frameNo),' ','')]);
    % fname3 = fname2 + ".csv";
    % myTable2 = struct2table(myStruct2);
    % writetable(myTable2,fname3,'delimiter',',');
    
    
    % Save IQ
    label = '802.11n_IQ';
    fname3 = fullfile(trainDir, [label '_frame_' strrep(num2str(frameNo),' ','')]);
    % write_binary (fname, 2, waveform)
    save(fname3 + ".mat", 'waveform')
    
    fileID = fopen(fname3 + ".yaml",'w');
         metadata = strcat('# ---------------------------------------------------------------------,\n', ...
            '# Waveform Configuration]\n', ...
            '# ---------------------------------------------------------------------,]\n', ...
            'standard: "802.11n_matlab"\n', ...
            'modulation: "OFDM"\n', ... 
            'MCS: "16-QAM(1/2 rate)"\n', ... %add a flag to auto change for 1/2 vs 3/4 rate, automate this
            'bandwidth: ' + " " ,num2str(Fs), '\n', ... 
            'sampling_rate: ' + " " ,num2str(Fs), '\n', ...
            'PSDU_length_bytes: 1035');   
         %             'time_duration: ' + " ", num2str(frameDuration), '\n', ... %is this needed ?

        %add other MIMO specific things? maybe create another category for the
        %MIMO related WIFI in wireless_link_paramter.yaml    
        fprintf(fileID, metadata);
        fclose(fileID);
end

for frameNo = 251:500 % first 1000 is 1/2 rate , 2nd 1000 is 3/4 rate

    vhtCfg = wlanVHTConfig('ChannelBandwidth', 'CBW20', ...
    'NumUsers', 1, ...
    'NumTransmitAntennas', 1, ...
    'NumSpaceTimeStreams', [1], ...
    'SpatialMapping', 'Direct', ...
    'STBC', false, ...
    'MCS', 4, ...
    'ChannelCoding', 'BCC', ...
    'APEPLength', 1024, ...
    'GuardInterval', 'Long', ...
    'GroupID', 63, ...
    'PartialAID', 275);
    
    
    numPackets = 6;
    % input bit source:
    in = randi([0, 1], 1000, 1);
    
    
    % Generation
    waveform = wlanWaveformGenerator(in, vhtCfg, ...
    'NumPackets', numPackets, ...
    'IdleTime', 0, ...
    'OversamplingFactor', 1, ...
    'ScramblerInitialization', 93, ...
    'WindowTransitionTime', 1e-07);

    Fs = wlanSampleRate(vhtCfg, 'OversamplingFactor', 1); 	
    
    
   
    
    %% Visualize
%     Spectrum Analyzer
%     spectrum = dsp.SpectrumAnalyzer('SampleRate', Fs, ...
%         'NumInputPorts',1, ...
%         'ShowLegend',true);
%     spectrum(waveform); %both
%     release(spectrum);
    
    %create spectrogram and PSD 
%     saveSpectrogramImage_Vini_RxScript(waveform,20e6,0,[20e6],'/home/genesys/Downloads/Stratis_Kaushik_Proposal/dataset1_0/802_11n/specs','802.11n_IQ',[256 256],1);   
%  %TO DO: REMOVE PSD PART IN PLOTS ??    
% rxWave,sr,fc,BW, folder,label,imageSize, idx (8)

    
    %% SAVE WAVEFORM
    
   

    % % Metadata File
    % label = '802.11g_Metadata';
    % fname2 = fullfile(trainDir, [label '_frame_' strrep(num2str(frameNo),' ','')]);
    % fname3 = fname2 + ".csv";
    % myTable2 = struct2table(myStruct2);
    % writetable(myTable2,fname3,'delimiter',',');
    
    
    % Save IQ
    label = '802.11n_IQ';
    fname3 = fullfile(trainDir, [label '_frame_' strrep(num2str(frameNo),' ','')]);
    % write_binary (fname, 2, waveform)
    save(fname3 + ".mat", 'waveform')
    
    fileID = fopen(fname3 + ".yaml",'w');
         metadata = strcat('# ---------------------------------------------------------------------,\n', ...
            '# Waveform Configuration]\n', ...
            '# ---------------------------------------------------------------------,]\n', ...
            'standard: "802.11n_matlab"\n', ...
            'modulation: "OFDM"\n', ... 
            'MCS: "16-QAM(3/4 rate)"\n', ... %add a flag to auto change for 1/2 vs 3/4 rate, automate this
            'bandwidth: ' + " " ,num2str(Fs), '\n', ... 
            'sampling_rate: ' + " " ,num2str(Fs), '\n', ...
            'PSDU_length_bytes: 1035');  
         %             'time_duration: ' + " ", num2str(frameDuration), '\n', ... %is this needed ?

        %add other MIMO specific things? maybe create another category for the
        %MIMO related WIFI in wireless_link_paramter.yaml    
        fprintf(fileID, metadata);
        fclose(fileID);
end