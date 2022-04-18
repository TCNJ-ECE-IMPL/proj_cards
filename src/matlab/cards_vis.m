close all
clear

figure('Position',[100 100 1800 800])

checkpoint = 'net_checkpoint__434__2022_04_18__11_09_25';

imds = imageDatastore('C:\Users\pearlstl\Downloads\cards_sparse_45deg_each_2deg\', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

filename = sprintf('C:\\Users\\pearlstl\\Documents\\_Teaching\\ELC_435\\MATLAB\\Checkpoints\\%s.mat',checkpoint);
load(filename);

% 23 rotations per card x 4 suits = 92 card offset between values
CARD_TO_CARD_OFFSET = 23 * 4;
acts = zeros(13*CARD_TO_CARD_OFFSET,32);
colors=zeros(13*4,1);
for card_idx=0:13*CARD_TO_CARD_OFFSET-1
    fn = imds.Files(card_idx+1);
    fn = fn{1};
    im = imread(fn);
    
    act1 = activations(net,im,'fc1');
    sz = size(act1);
%    acts(card_idx + 1,:) = squeeze(max(act1,[],[1,2]));
    acts(card_idx + 1,:) = squeeze(act1); % for fully connected
    act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    colors(card_idx + 1) = floor(card_idx/23);
%    I = imtile(mat2gray(act1),'GridSize',[4 8]);
%    I = imresize(I,1900/(sz(2)*8));
end

ts=tsne(acts,'NumDimensions',3);
ts2=tsne(acts,'NumDimensions',2);
figure(1)
colors = floor((0:13*CARD_TO_CARD_OFFSET-1)/(23*4));
% scatter3(ts(:,1),ts(:,2),ts(:,3),8,colors+1,'filled');
colormap('jet');

card_names = {'t','2','3','4','5','6','7','8','9','a','j','k','q'};
suit_names = {'c','d','h','s'};
all_cards={};
suit_colors=zeros(1,13*CARD_TO_CARD_OFFSET);
val_colors=zeros(1,13*CARD_TO_CARD_OFFSET);
for card_val_idx=0:12
    for rot_idx=0:22
        for suit_idx=0:3
            full_idx = card_val_idx*CARD_TO_CARD_OFFSET+rot_idx*4+suit_idx;
            suit_colors(full_idx+1) = suit_idx+1;
            val_colors(full_idx+1) = card_val_idx+1;
%            cs=sprintf('%s%s',card_names{card_val_idx+1},suit_names{suit_idx+1});
%             cs=sprintf('%s',card_names{card_val_idx+1});
            cs=sprintf('%d',rot_idx);
            all_cards{full_idx+1}=cs;
        end
    end
end

suit_colors=categorical(suit_colors);
val_colors=categorical(val_colors);

card_indices=floor((0:13*CARD_TO_CARD_OFFSET-1)/CARD_TO_CARD_OFFSET)+1;
%all_cards=card_names(card_indices);
%textscatter3(ts,all_cards,'TextDensityPercentage',90,'MarkerColor','auto','ColorData',suit_colors);

textscatter(ts2,all_cards,'TextDensityPercentage',100,'MarkerColor','auto','ColorData',suit_colors);
% 
% figure(2)
% channels = 1:32;
% name = 'conv'
% I = deepDreamImage(net,name,channels, 'PyramidLevels',7);
% 
% I = imtile(I,'ThumbnailSize',[128 128]);
% imshow(I)

% layer = 2;
% wgts = net.Layers(layer).Weights
% tiled = imtile(wgts,'GridSize',[4 8]);
% tiled=imresize(tiled,16,'Nearest');
% imshow(tiled)