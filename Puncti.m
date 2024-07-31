% Written by Damien Jullié, feel free to spread.
% Base structure based on Corona
% Code displays 3 channels
% Code allows to select ROIs based on a segmentation in one or two channels of choice
% Quantifies the background subtracted fluorescence at segmented structures



function Puncti(action)


if nargin == 0

versiontrack = {'07/19/2022'};


TC = '3';
MTC = '1200';
FT = '1';
FT2 = '1';




defaults = {TC,MTC,FT,FT2};
prompt = {'min size','max size','Fourier','Fourier Background'};
answer = inputdlg(prompt,'Parameters for quantification',1,defaults);


TC = str2double(answer{1});
MTC = str2double(answer{2});
FT = str2double(answer{3});
FT2 = str2double(answer{4});


parameters = [TC,MTC,FT,FT2];


[stk,stkd] = uigetfile('*.tif','Receptor images'); %this is gonna be the blue channel
info = imfinfo(stk);
num_images = numel(info);

[stk2,stkd2] = uigetfile('*.tif','Marker images'); %this is gonna be the green channel
info2 = imfinfo(stk2);
num_images2 = numel(info2);

[stk3,stkd3] = uigetfile('*.tif','Prot X images'); % this is gonna be the red channel
info3 = imfinfo(stk3);
num_images3 = numel(info3);

%write the parameters in the final excel file
Legend1 = {'version','min size','max size','Fourier','Fourier Back','Receptor','Marker','Prot X'};
Param = cat(2, versiontrack,num2cell(parameters),{stk,stk2,stk3});
Param = cat(1,Legend1,Param);
Param = {Param};


%initialize matrix for images to turn into movies

M = [];
red = [];
blue = [];

%initialize matrix for image names

  

    for k = 1:num_images2
        g = imread(stk2,k);
        M = cat(3,M,g);
    end

    for m = 1: num_images3
        r = imread(stk3,m);
        red = cat(3,red,r);
    end

    for n = 1: num_images
        b = imread(stk,n);
        blue = cat(3,blue,b);
    end



%at this stage loaded all images in 3 matrixes + name of cells in 3
%matrixes


   frame = 1;
   figure('name',['Puncti ',stk])

   %%%controls for playing the movie


   uicontrol('callback','Puncti goto','string','frame#','style','slider',...
      'position',[50,15,90,15],'max',size(M,3),'min',1,'value',1,...
      'sliderstep',[1/(size(M,3)-1),10/(size(M,3)-1)],'tag','frame#');
   uicontrol('style','text','position',[10,15,35,15],'string','Frame')
   uicontrol('style','text','position',[70,30,30,15],'tag','FrameTxt')

   %%%controls for the scale
   highG = double(max(max(max(M))));
   lowG = double(min(min(min(M))));
   highR = double(max(max(max(red))));
   lowR = double(min(min(min(red))));
   highB = double(max(max(max(blue))));
   lowB = double(min(min(min(red))));


   %Control Green

   uicontrol('style','slider','callback','Puncti scaleG',...
      'min',lowG,'max',highG-3,'value',lowG,...
      'position',[200,15,100,15],'tag','scalelowG')
   uicontrol('style','text','position',[310,15,30,15],'tag','low_textG')
   uicontrol('style','text','position',[160,15,35,15],'string','Low','tag','lowG','userdata',lowG)
   uicontrol('style','slider','callback','Puncti scaleG',...
      'min',lowG+3,'max',highG,'value',highG,...
      'position',[200,30,100,15],'tag','scalehighG')
   uicontrol('style','text','position',[310,30,30,15],'tag','high_textG')
   uicontrol('style','text','position',[160,30,35,15],'string','High','tag','highG','userdata',highG)

    
    %Control Red
   uicontrol('style','slider','callback','Puncti scaleR',...
      'min',lowR,'max',highR-3,'value',lowR,...
      'position',[380,15,100,15],'tag','scalelowR')
   uicontrol('style','text','position',[490,15,30,15],'tag','low_textR')
   uicontrol('style','text','position',[340,15,35,15],'string','Low','tag','lowR','userdata',lowR)
   uicontrol('style','slider','callback','Puncti scaleR',...
      'min',lowR+3,'max',highR,'value',highR,...
      'position',[380,30,100,15],'tag','scalehighR')
   uicontrol('style','text','position',[490,30,30,15],'tag','high_textR')
   uicontrol('style','text','position',[340,30,35,15],'string','High','tag','highR','userdata',highR)

    %Control Blue
      uicontrol('style','slider','callback','Puncti scaleB',...
      'min',lowB,'max',highB-3,'value',lowB,...
      'position',[560,15,100,15],'tag','scalelowB')
   uicontrol('style','text','position',[670,15,30,15],'tag','low_textB')
   uicontrol('style','text','position',[520,15,35,15],'string','Low','tag','lowB','userdata',lowB)
   uicontrol('style','slider','callback','Puncti scaleB',...
      'min',lowB+3,'max',highB,'value',highB,...
      'position',[560,30,100,15],'tag','scalehighB')
   uicontrol('style','text','position',[670,30,30,15],'tag','high_textB')
   uicontrol('style','text','position',[520,30,35,15],'string','High','tag','highB','userdata',highB)


   %%% Controls for detection and track of objects

   coord = {};


   uicontrol('string','Select ROI','position',[30,250,80,30],...
      'callback','Puncti Box','tag','coord','userdata',coord)
   uicontrol('string','Export Data','position',[30,200,80,30],...
      'callback','Puncti Quant','tag','Quantax','userdata',Param)
  

   %%%controls for the zoom button
   uicontrol('style','toggle','position',[40,345,60,30],'string','ZOOM',...
       'tag','zoomOn','callback','Puncti zoomToggle','value',0)

   
   
   rgb = cat(3,red(:,:,frame),M(:,:,frame),blue(:,:,frame));
   rgbmovie = {double(M),double(red),double(blue)};
   u = image(rgb,'cdatamapping','direct','tag','movi','userdata',rgbmovie);
   

   set(gca,'position',[0.13,0.15,0.9,0.75])
   h = title([stk,' Frame # = ',num2str(frame)],...
      'interpreter','none');
   set(h,'userdata',stk)
   axis image
   
   scaleG
   scaleR
   scaleB
   goto
else
  eval(action)
end

function zoomToggle
children = get(gcf,'children');
zoomStatus = get(findobj(children,'tag','zoomOn'),'value');
if zoomStatus
    set(findobj(children,'tag','zoomOn'),'ForegroundColor','r')
    set(findobj(children,'tag','selectR'),'value',0)
    zoom on

else
    set(findobj(children,'tag','zoomOn'),'ForegroundColor','default')
    zoom off
end

function scaleG
children = get(gcf,'children');
lowG = round(get(findobj(children,'tag','scalelowG'),'value'));
highG = round(get(findobj(children,'tag','scalehighG'),'value'));

minlow = round(get(findobj(children,'tag','lowG'),'userdata'));
maxhigh = round(get(findobj(children,'tag','highG'),'userdata'));

if highG > maxhigh
    highG = maxhigh;
end
if lowG < minlow
    lowG = minlow;
end
if highG == minlow+1
   highG = highG +1;
end
if lowG == maxhigh-1
   lowG = lowG-1;
end


set(findobj(children,'tag','scalelowG'),'max',highG-1,'min',minlow,...
   'sliderstep',[1/(highG-1-minlow),25/(highG-1-minlow)],...
   'value',lowG)
set(findobj(children,'tag','low_textG'),'string',num2str(lowG));
set(findobj(children,'tag','scalehighG'),'min',lowG+1,'max',maxhigh,...
   'sliderstep',[1/(maxhigh-(lowG+1)),25/(maxhigh - (lowG+1))],...
   'value',highG)
set(findobj(children,'tag','high_textG'),'string',num2str(highG));

scaleMerge

function scaleR

children = get(gcf,'children');
lowR = round(get(findobj(children,'tag','scalelowR'),'value'));
highR = round(get(findobj(children,'tag','scalehighR'),'value'));

minlow = round(get(findobj(children,'tag','lowR'),'userdata'));
maxhigh = round(get(findobj(children,'tag','highR'),'userdata'));


if highR > maxhigh
    highR = maxhigh;
end
if lowR < minlow
    lowR = minlow;
end
if highR == minlow+1
   highR = highR +1;
end
if lowR == maxhigh-1
   lowR = lowR-1;
end


set(findobj(children,'tag','scalelowR'),'max',highR-1,'min',minlow,...
   'sliderstep',[1/(highR-1-minlow),25/(highR-1-minlow)],...
   'value',lowR)
set(findobj(children,'tag','low_textR'),'string',num2str(lowR));
set(findobj(children,'tag','scalehighR'),'min',lowR+1,'max',maxhigh,...
   'sliderstep',[1/(maxhigh-(lowR+1)),25/(maxhigh - (lowR+1))],...
   'value',highR)
set(findobj(children,'tag','high_textR'),'string',num2str(highR));

scaleMerge

function scaleB

children = get(gcf,'children');
lowB = round(get(findobj(children,'tag','scalelowB'),'value'));
highB = round(get(findobj(children,'tag','scalehighB'),'value'));

minlow = round(get(findobj(children,'tag','lowB'),'userdata'));
maxhigh = round(get(findobj(children,'tag','highB'),'userdata'));


if highB > maxhigh
    highB = maxhigh;
end
if lowB < minlow
    lowB = minlow;
end
if highB == minlow+1
   highB = highB +1;
end
if lowB == maxhigh-1
   lowB = lowB-1;
end


set(findobj(children,'tag','scalelowB'),'max',highB-1,'min',minlow,...
   'sliderstep',[1/(highB-1-minlow),25/(highB-1-minlow)],...
   'value',lowB)
set(findobj(children,'tag','low_textB'),'string',num2str(lowB));
set(findobj(children,'tag','scalehighB'),'min',lowB+1,'max',maxhigh,...
   'sliderstep',[1/(maxhigh-(lowB+1)),25/(maxhigh - (lowB+1))],...
   'value',highB)
set(findobj(children,'tag','high_textB'),'string',num2str(highB));

scaleMerge

function scaleMerge
children = get(gcf,'children');
mergeHandle = findobj(children,'tag','movi');
rgb = get(findobj(children,'tag','movi'),'userdata');
frame = round(get(findobj(children,'tag','frame#'),'value'));
G = rgb{1};
imG = double(G(:,:,frame));
R = rgb{2};
imR = double(R(:,:,frame));
B = rgb{3};
imB = double(B(:,:,frame));


lowG = get(findobj(children,'tag','scalelowG'),'value');
highG = get(findobj(children,'tag','scalehighG'),'value');
lowR = get(findobj(children,'tag','scalelowR'),'value');
highR = get(findobj(children,'tag','scalehighR'),'value');
lowB = get(findobj(children,'tag','scalelowB'),'value');
highB = get(findobj(children,'tag','scalehighB'),'value');

green = (imG-lowG)/(highG-lowG);
green(green<0) = 0;
green(green>1) = 1;
red = (imR-lowR)/(highR-lowR);
red(red<0) = 0;
red(red>1) = 1;
blue = (imB-lowB)/(highB-lowB);
blue(blue<0) = 0;
blue(blue>1) = 1;



rgb = cat(3,red,green,blue);
set(mergeHandle,'cdata',rgb)


function goto

children = get(gcf,'children');
rgb = get(findobj(children,'tag','movi'),'userdata');
M = rgb{1};
red = rgb{2};
blue = rgb{3};
frame = round(get(findobj(children,'tag','frame#'),'value'));
coord = get(findobj(children,'tag','coord'),'userdata');

delete(findobj(children,'type','line','color','red'))
delete(findobj(children,'type','line','color',[1 1 1]))

if ~isempty(coord)
    
    back = coord{1,1};
    back = back(1,2:end);
    bx = back(1); by = back(2); bw = back(3); bh = back(4);
    X = [bx,bx,bx+bw,bx+bw,bx];
    Y = [by,by+bh,by+bh,by,by];
    line('XData',X,'YData',Y,'color','r')
    
    if size(coord,2)>1
        for i = 2:size(coord,2)
        posPoly = coord(1,i);
        posPoly = posPoly{1,1};
        posPoly = posPoly(6:end,:);
        line(cat(1,posPoly(:,1),posPoly(1,1)),cat(1,posPoly(:,2),posPoly(1,2)),'lineStyle','-','marker','none','Color',[1 1 1],'LineWidth',2)
        end
    end
end
    

global stop
stop = 1;


img = cat(3,red(:,:,frame),M(:,:,frame),blue(:,:,frame));
set(findobj(children,'tag','FrameTxt'),'string',num2str(frame));
set(findobj(children,'tag','movi'),'cdata',img)
tit = get(gca,'title');
stk = get(tit,'userdata');
title(stk);
scaleMerge



function Box     %draw region

children = get(gcf,'children');
mergeHandle = findobj(children,'tag','movi');
rgb = get(findobj(children,'tag','movi'),'userdata');
frame = round(get(findobj(children,'tag','frame#'),'value'));
coord = get(findobj(children,'tag','coord'),'userdata');
parameters = get(findobj(children,'tag','Quantax'),'userdata');
M = rgb{1};
red = rgb{2};
blue = rgb{3};


tit = get(gca,'title');
stk = get(tit,'userdata');


if isempty(coord)

msgbox('Select the background region')
waitfor(gcf)

    [Xback,Yback,Back,rect] = imcrop;
    rect = round(rect);
    bx = rect(1); by = rect(2); bw = rect(3); bh = rect(4);
    X = [bx,bx,bx+bw,bx+bw,bx];
    Y = [by,by+bh,by+bh,by,by];
    line('XData',X,'YData',Y,'color','r')
    Coordrect = cat(2,0,rect);
    coord = {Coordrect;''};
    
end

polygon = impoly; % draw polygon around ROI
cellMask = createMask(polygon);
posPoly = getPosition(polygon);

%get bounding box with 20 pix margin
minP = min(posPoly,[],1);
maxP = max(posPoly,[],1);

minP = floor(minP-20);
minP = (minP > 0) .* minP;
minP = minP + 1;

maxP = ceil(maxP+20);
maxP = ((maxP > 512) .* [512,512] + (maxP < 512) .* maxP) ;
maxP = maxP -1;

BBox = cellMask(minP(1,2):maxP(1,2),minP(1,1):maxP(1,1));

%I know it looks scary but this is just drawing some fuckin lines around
%the polygon you just draw
line(cat(1,posPoly(:,1),posPoly(1,1)),cat(1,posPoly(:,2),posPoly(1,2)),'lineStyle','-','marker','none','Color',[1 1 1],'LineWidth',2)

%save the coordinates of bounding box into the polygon
posPoly = cat(1,minP,maxP,posPoly);

coordPoly = {posPoly;''};
coord = cat(2, coord, coordPoly);
set(findobj(children,'tag','coord'),'userdata',coord)

delete(polygon)

miniM = M(minP(1,2):maxP(1,2),minP(1,1):maxP(1,1),:);
miniR = red(minP(1,2):maxP(1,2),minP(1,1):maxP(1,1),:);
miniB = blue(minP(1,2):maxP(1,2),minP(1,1):maxP(1,1),:);

frMskG = miniM(:,:,frame); % green mask made on current frame

%frMskG = miniM(:,:,max(1,frame - 2):min(frame + 2,size(miniM,3))); %green mask made on 5 frames average
%frMskG = mean(frMskG,3);
frMskR = miniR(:,:,frame); % red mask made on current frame
frMskB = miniB(:,:,frame);

lowR = min(min(min(miniR,[],3)));
highR = max(max(max(miniR,[],3)));

lowG = min(min(min(miniM,[],3)));
highG = max(max(max(miniM,[],3)));

CThR = round(lowR + (highR-lowR)/5);
CThG = round(lowG + (highG-lowG)/5);


mskMapG = gray(256);
mskMapR = gray(256);
mskMapR(end,:) = [1 0 0];
mskMapG(end,:) = [0 1 0];
mskMapF = [0,0,0;1,0,0;0,1,0;1,1,0];

figure('name',['Mask ',stk])

%Controls to play the movie

uicontrol('callback','Puncti scaleMsk','string','frame#','style','slider',...
      'position',[50,20,90,15],'max',size(M,3),'min',1,'value',frame,...
      'sliderstep',[1/(size(M,3)-1),10/(size(M,3)-1)],'tag','frameMask');
uicontrol('style','text','position',[10,20,35,15],'string',num2str(frame),'tag','Param','userdata',parameters)

%control for receptor movie

uicontrol('callback','Puncti scaleMsk','string','frame#','style','slider',...
      'position',[220,10,200,15],'max',10,'min',1,'value',1,...
      'sliderstep',[1/10,1/5],'tag','NReg');
uicontrol('style','text','position',[150,10,50,15],'string','NReg','tag','NRegvalue')



uicontrol('position',[220,30,200,15],'style','slider',...
    'callback','Puncti scaleMsk','tag','scalethR',...
    'min',0,'max',highR,'value',CThR, 'sliderstep',[1/10000,1/100])
uicontrol('position',[150,30,50,15],'style','text','string','Thresh R','tag','miniR','userdata',miniR)

% controls for the Marker movie


uicontrol('position',[520,30,200,15],'style','slider',...
    'callback','Puncti scaleMsk','tag','FFTG',...
    'min',0,'max',10,'value',1,'sliderstep',[1/50,1/10])
uicontrol('position',[450,30,50,15],'style','text','string','FFTG','tag','miniM','userdata',miniM)

uicontrol('position',[520,10,200,15],'style','slider',...
    'callback','Puncti scaleMsk','tag','FFTG2',...
    'min',0,'max',50,'value',3,'sliderstep',[1/50,1/5])
uicontrol('position',[450,10,50,15],'style','text','string','FFTG2','tag','FFTG2s','userdata',miniB)

uicontrol('position',[520,50,200,15],'style','slider',...
    'callback','Puncti scaleMsk','tag','scalethG',...
    'min',0,'max',highG,'value',CThG, 'sliderstep',[1/10000,1/100])
uicontrol('position',[450,50,50,15],'style','text','string','Thresh G')


% controls for the cellshape movie
uicontrol('position',[220,50,200,15],'style','slider',...
    'callback','Puncti scaleMsk','tag','Shape',...
    'min',0,'max',round(max(max(frMskB))),'value',100,'sliderstep',[1/10000,1/100])
uicontrol('position',[150,50,50,15],'style','text','string','CellMask')


% controls for dialogue
uicontrol('position',[20,150,90,20],'string','Create Mask',...
    'callback','Puncti createMiniMsk','tag','Export')
uicontrol('position',[20,180,90,20],'string','Edit Poly',...
    'callback','Puncti ExitBox','tag','BoxR','userdata',BBox)

subplot(1,3,2,'replace')
image(frMskG,'cdatamapping','scaled','tag','maskImage')
colormap(gca,mskMapR)
title('Marker')
axis image

subplot(1,3,1,'replace')
image(frMskR,'cdatamapping','scaled');
colormap(gca,mskMapG)
title('Protein X')
axis image

subplot(1,3,3,'replace')
image(BBox,'cdatamapping','scaled');
colormap(gca,mskMapF)
title('Mask')
axis image

scaleMsk


waitfor(gcf)

goto

function scaleMsk
children = get(gcf,'children');
%DilR = round(get(findobj(children,'tag','dilR'),'value'));
FFTG = get(findobj(children,'tag','FFTG'),'value');
FFTG2 = round(get(findobj(children,'tag','FFTG2'),'value'));
threshR = round(get(findobj(children,'tag','scalethR'),'value'));
threshG = round(get(findobj(children,'tag','scalethG'),'value'));
lowR = get(findobj(children,'tag','scalethR'),'min');
highR = get(findobj(children,'tag','scalethR'),'max');
lowG = get(findobj(children,'tag','scalethG'),'min');
highG = get(findobj(children,'tag','scalethG'),'max');
frame = round(get(findobj(children,'tag','frameMask'),'value'));
%rgb = get(findobj(children,'tag','movi2'),'userdata');
%coord = get(findobj(children,'tag','coord'),'userdata');
parameters = get(findobj(children,'tag','Param'),'userdata');
miniR = get(findobj(children,'tag','miniR'),'userdata');
miniG = get(findobj(children,'tag','miniM'),'userdata');
NReg = round(get(findobj(children,'tag','NReg'),'value'));
BBox = get(findobj(children,'tag','BoxR'),'userdata');
shape = round(get(findobj(children,'tag','Shape'),'value'));
miniB = get(findobj(children,'tag','FFTG2s'),'userdata');

Param = parameters{1,1};
Objsize = Param{2,2};
Msize = Param{2,3};
FT = Param{2,4};
FT2 = Param{2,5};

frMskR = miniR(:,:,frame);
frMskG = miniG(:,:,frame);
%frMskG = miniG(:,:,max(1,frame - 2):min(frame + 2,size(miniG,3))); %green mask made on 5 frames average
%frMskG = mean(frMskG,3);
frMskB = max(miniB,[],3);


%Segment cellshape
cellshape = frMskB >= shape;
BBox = cellshape & BBox;

seG = strel('disk',2,4);

BBox = imdilate(BBox,seG);
BBox = imerode(BBox,seG);

BBox = imerode(BBox,seG);
BBox = imdilate(BBox,seG);

bwBox = bwlabel(BBox,4);
propR = regionprops(bwBox,'Area');
SRR = cat(1,[propR.Area],1:size([propR.Area],2))';
SRR = sortrows(SRR,1);

BBox = bwBox == SRR(end,2);

%segment green marker with double fourrier

if FT ==1

b = fft2(frMskG);
D2 = fftshift(b);


M = ones(size(frMskG));
I1 = 1:size(frMskG,1);
I2 = 1:size(frMskG,2);
x = I2-size(frMskG,2)/2;
y = I1-size(frMskG,1)/2;
[X,Y] = meshgrid(x,y);
A = (X.^2 + Y.^2 <= FFTG^2);
M1 = M;
M1(A) = 0;
D3 = M1.*D2;
D4 = ifftshift(D3);
D5 = ifft2(D4);
D6 = abs(D5);

if FT2 == 1

b2 = fft2(frMskG);
D22 = fftshift(b2);
A = (X.^2 + Y.^2 >= FFTG2^2);
M2 = M;
M2(A) = 0;
D32 = M2 .*D22;
D42 = ifftshift(D32);
D52 = ifft2(D42);
D62 = abs(D52);


%cst = min(min(D6-D62));

%if cst < 0
%    cst = -cst;
%else
%    cst = 0;
%end

cst = 1000;

finalFT = D6-D62 + cst;

D7 = finalFT > threshG;

else

D7 = D6 > threshG ;
end

else
    
D7 = frMskG > threshG;

end
MaskG = D7 > 0;

%Smooth
seG = strel('disk',1,4);

MaskG = imerode(MaskG,seG);
MaskG = imdilate(MaskG,seG);

MaskG = imdilate(MaskG,seG);
MaskG = imerode(MaskG,seG);

%size exclusion
MaskG = bwlabel(MaskG,4);

for caca = 1:max(max(MaskG));
    
    Mask2 = MaskG == caca;
    
    if sum(sum(Mask2)) < Objsize  %%% here is minimal size of the objects

    MaskG(find(MaskG==caca))=0;


    end
    
    if sum(sum(Mask2)) > Msize

    MaskG(find(MaskG==caca))=0;
    
    end
end

MaskG = MaskG > 0;

finalG = BBox & MaskG;

%segment receptor max projection
MaskR = frMskR > threshR;
MaskR = MaskR > 0;
MaskR = BBox & MaskR;

%keep the NReg biggest regions

seG = strel('disk',1,4);

MaskR = imerode(MaskR,seG);
MaskR = imdilate(MaskR,seG);

MaskR = imdilate(MaskR,seG);
MaskR = imerode(MaskR,seG);

MaskBW = bwlabel(MaskR,4);

if max(max(MaskBW))> NReg

propR = regionprops(MaskBW,'Area');
SRR = cat(1,[propR.Area],1:size([propR.Area],2))';
SRR = sortrows(SRR,1);

MaskRNReg = zeros(size(MaskR));

for i = 1:NReg;

MaskRNReg = MaskRNReg | (MaskBW == SRR(end -i +1,2));

end

finalR = MaskRNReg > 0;
imR = frMskR - lowR;
imR = imR./threshR;
imR(imR >1) = 1;
imR = imR./1.01;
imR(finalR) = 1;
else
imR = frMskR - lowR;
imR = imR./threshR;
imR(imR >1) = 1;
imR = imR./1.01;
imR(MaskR) = 1;
finalR = MaskR > 0;
    
end

mskMapG = gray(256);
mskMapR = gray(256);
mskMapR(end,:) = [1 0 0];
mskMapG(end,:) = [0 1 0];
mskMapF = [0,0,0;0,0,0.2;0.5,0,0;0,1,0];

%use threshold R for colormap on max receptor proj



subplot(1,3,1,'replace')
image(imR,'cdatamapping','scaled');
caxis([0,1]);
axis image
colormap(gca,mskMapR)
title('Protein X')


%use threshold G for colormap on current marker frame


imG = frMskG - lowG;
imG = imG./(threshG*5);
imG(imG>1) = 1;
imG = imG./1.01;
imG(finalG) = 1;

subplot(1,3,2,'replace')
image(imG,'cdatamapping','scaled');
caxis([0,1]);
axis image
colormap(gca,mskMapG)
title('Marker')

%combine masks

finalmask = finalR & finalG;

imMask = BBox + finalR + finalmask;

subplot(1,3,3,'replace')
image(imMask,'cdatamapping','scaled');
caxis([0,4]);
axis image
colormap(gca,mskMapF)
title('Mask')

set(findobj(children,'tag','miniM'),'string',['FFTG ',num2str(FFTG)])
set(findobj(children,'tag','NRegvalue'),'string',['NReg ',num2str(NReg)])
set(findobj(children,'tag','FFTG2s'),'string',['FFTG2 ',num2str(FFTG2)])
set(findobj(children,'tag','Param'),'string',num2str(frame))
final = [threshG, threshR, NReg,FFTG,FFTG2];
set(findobj(children,'tag','Export'),'userdata',final)

function createMiniMsk

children = get(gcf,'children');
final = get(findobj(children,'tag','Export'),'userdata');
parameters = get(findobj(children,'tag','Param'),'userdata');
miniR = get(findobj(children,'tag','miniR'),'userdata');
miniG = get(findobj(children,'tag','miniM'),'userdata');
miniB = get(findobj(children,'tag','FFTG2s'),'userdata');
BBoxRef = get(findobj(children,'tag','BoxR'),'userdata');
shape = round(get(findobj(children,'tag','Shape'),'value'));

Param = parameters{1,1};
Objsize = Param{2,2};
Msize = Param{2,3};
FT = Param{2,4};
FT2 = Param{2,5};


threshG = final(1,1);
threshR = final(1,2);
NReg = final(1,3);
FFTG = final(1,4);
FFTG2 = final(1,5);

moviemask = [];

%Segment cellshape
frMskB = max(miniB,[],3);
cellshape = frMskB >= shape;
BBox = cellshape & BBoxRef;

seG = strel('disk',2,4);

BBox = imdilate(BBox,seG);
BBox = imerode(BBox,seG);

BBox = imerode(BBox,seG);
BBox = imdilate(BBox,seG);

bwBox = bwlabel(BBox,4);
propR = regionprops(bwBox,'Area');
SRR = cat(1,[propR.Area],1:size([propR.Area],2))';
SRR = sortrows(SRR,1);

BBox = bwBox == SRR(end,2);

for j = 1:size(miniG,3)
    
    frMskG = miniG(:,:,j);
    %frMskG = miniG(:,:,max(1,j - 2):min(j + 2,size(miniG,3))); %green mask made on 5 frames average
    %frMskG = mean(frMskG,3);
    frMskR = miniR(:,:,j);

%segment green marker with double fourrier

if FT ==1

b = fft2(frMskG);
D2 = fftshift(b);


M = ones(size(frMskG));
I1 = 1:size(frMskG,1);
I2 = 1:size(frMskG,2);
x = I2-size(frMskG,2)/2;
y = I1-size(frMskG,1)/2;
[X,Y] = meshgrid(x,y);
A = (X.^2 + Y.^2 <= FFTG^2);
M1 = M;
M1(A) = 0;
D3 = M1.*D2;
D4 = ifftshift(D3);
D5 = ifft2(D4);
D6 = abs(D5);

if FT2 == 1

b2 = fft2(frMskG);
D22 = fftshift(b2);
A = (X.^2 + Y.^2 >= FFTG2^2);
M2 = M;
M2(A) = 0;
D32 = M2 .*D22;
D42 = ifftshift(D32);
D52 = ifft2(D42);
D62 = abs(D52);

%cst = min(min(D6-D62));

%if cst < 0
%    cst = -cst;
%else
%    cst = 0;
%end

cst = 1000;
finalFT = D6-D62 + cst;

D7 = finalFT > threshG;

else

D7 = D6 > threshG ;
end

else
    
D7 = frMskG > threshG;

end
MaskG = D7 > 0;

%Smooth

seG = strel('disk',1,4);

MaskG = imerode(MaskG,seG);
MaskG = imdilate(MaskG,seG);

MaskG = imdilate(MaskG,seG);
MaskG = imerode(MaskG,seG);


%size exclusion
MaskG = bwlabel(MaskG,4);


for caca = 1:max(max(MaskG));
    
    Mask2 = MaskG == caca;
    
    if sum(sum(Mask2)) < Objsize  %%% here is minimal size of the objects

    MaskG(find(MaskG==caca))=0;

    end
    
        if sum(sum(Mask2)) > Msize

    MaskG(find(MaskG==caca))=0;
    
    end
end

MaskG = MaskG > 0;

finalG = BBox & MaskG;

%segment receptor max projection
MaskR = frMskR > threshR;
MaskR = MaskR > 0;
MaskR = BBox & MaskR;

%Smooth n keep the biggest regions
seG = strel('disk',1,4);

MaskR = imerode(MaskR,seG);
MaskR = imdilate(MaskR,seG);

MaskR = imdilate(MaskR,seG);
MaskR = imerode(MaskR,seG);

MaskBW = bwlabel(MaskR,4);

if max(max(MaskBW))> NReg

propR = regionprops(MaskBW,'Area');
SRR = cat(1,[propR.Area],1:size([propR.Area],2))';
SRR = sortrows(SRR,1);

MaskRNReg = zeros(size(MaskR));

for i = 1:NReg;

MaskRNReg = MaskRNReg | (MaskBW == SRR(end -i +1,2));

end

finalR = MaskRNReg > 0;

else

finalR = MaskR > 0;

end

%combine masks

finalmask = finalR & finalG;

imMask = BBox + finalR + finalmask;

moviemask = cat(3,moviemask,imMask);

end

moviemask = cat(4,moviemask,miniG,miniR,miniB);

close(gcf)
coord = get(findobj(gcf,'tag','coord'),'userdata');
posPoly = coord(:,end);
final = cat(1,[final(1,1),final(1,2)],[final(1,3),final(1,4)],[final(1,5),0]);
posPoly{1,1} = cat(1,final,posPoly{1,1});
posPoly{2,1} = moviemask;
coord(:,end) = posPoly;
set(findobj(gcf,'tag','coord'),'userdata',coord)

function ExitBox

close(gcf)
coord = get(findobj(gcf,'tag','coord'),'userdata');
coord = coord(:,1:end-1);
set(findobj(gcf,'tag','coord'),'userdata',coord)
goto


function Quant

children = get(gcf,'children');
coord = get(findobj(children,'tag','coord'),'userdata');
parameters = get(findobj(children,'tag','Quantax'),'userdata');
tit = get(gca,'title');
stk = get(tit,'userdata');
rgb = get(findobj(children,'tag','movi'),'userdata');
G = rgb{1};
R = rgb{2};
B = rgb{3};


 

%calculate background Fluo
backcoord = coord{1,1};
bx = backcoord(2);
by = backcoord(3);
bw = backcoord(4);
bh = backcoord(5);

BackG = sum(sum(G(by:by+bh,bx:bx+bw,:)))/((1+bw)*(1+bh));
BackG = squeeze(BackG)';
BackR = sum(sum(R(by:by+bh,bx:bx+bw,:)))/((1+bw)*(1+bh));
BackR = squeeze(BackR)';
BackB = sum(sum(B(by:by+bh,bx:bx+bw,:)))/((1+bw)*(1+bh));
BackB = squeeze(BackB)';


%loop through ROIs

ROIinfo = parameters{:,:};
ROIinfo = cat(1,ROIinfo,cell(1,8));
    
for i = 2:size(coord,2)


    %write parameters sheet

    final = coord{1,i};
    legendpos = {'Thresh Marker/Prot X';'NReg/FFTG';'FFTG2'};
    legendpos = cat(2,legendpos,num2cell(final(1:3,:)));
    ROICol = cell(size(final(4:end,:),1),1);
    ROICol(:,:) = {i-1};
    PosPoly = cat(2,ROICol,num2cell(final(4:end,:)));
    PosPoly = cat(1,legendpos,PosPoly);
    PosPoly = cat(2,PosPoly,cell(size(PosPoly,1),5));
    ROIinfo = cat(1,ROIinfo, PosPoly);
    
    %Extract ROIs variables
    coordReg = coord(:,i);
    Thresh = coordReg{1,1};
    catmovies = coordReg{2,1};
    moviemask = catmovies(:,:,:,1);
    miniG = catmovies(:,:,:,2);
    miniR = catmovies(:,:,:,3);
    miniB = catmovies(:,:,:,4);
    
    %get that final mask generated
    minP = Thresh(4,:);
    maxP = Thresh(5,:);
    
    
    %submasks by values
    maskB = moviemask > 0;
    maskR = moviemask == 2 | moviemask == 3;
    maskG = moviemask ==3;
    
    
    
    %R fluorescence in maskR
    
    fluoRR = maskR.* miniR;
    fluoRR = sum(sum(fluoRR))./sum(sum(maskR));
    fluoRR = squeeze(fluoRR)';
    fluoRR = fluoRR - BackR;
    
    %G fluorescence in maskR
    
    fluoGR = maskR.* miniG;
    fluoGR = sum(sum(fluoGR))./sum(sum(maskR));
    fluoGR = squeeze(fluoGR)';
    fluoGR = fluoGR - BackG;    
    
    %B fluorescence in maskR
    
    fluoBR = maskR.* miniB;
    fluoBR = sum(sum(fluoBR))./sum(sum(maskR));
    fluoBR = squeeze(fluoBR)';
    fluoBR = fluoBR - BackB;
    
    %G fluorescence in maskG
    
    fluoGG = maskG.* miniG;
    fluoGG = sum(sum(fluoGG))./sum(sum(maskG));
    fluoGG = squeeze(fluoGG)';
    fluoGG = fluoGG - BackG;
    
    %R fluorescence in maskG
    
    fluoRG = maskG.* miniR;
    fluoRG = sum(sum(fluoRG))./sum(sum(maskG));
    fluoRG = squeeze(fluoRG)';
    fluoRG = fluoRG - BackR;
    
    %B fluorescence in maskG
    
    fluoBG = maskG.* miniB;
    fluoBG = sum(sum(fluoBG))./sum(sum(maskG));
    fluoBG = squeeze(fluoBG)';
    fluoBG = fluoBG - BackB;
    
    
    Fluo = cat(1,fluoRR,fluoGR,fluoBR,fluoGG,fluoRG,fluoBG);
    



legend = cell(size(Fluo,1)+1,size(Fluo,2)+1);
legend(:,1) = {'frame';'ProtX@ProtX';'Marker@ProtX';'Receptor@ProtX';'Marker@Marker';'ProtX@Marker';'Receptor@Marker'};
legend(1,2:end) = num2cell([1:size(Fluo,2)]);
legend(2:end,2:end) = num2cell(Fluo);


sheet = array2table(legend);

writetable(sheet,[stk,'_Puncti.xlsx'],'Sheet',['Region ',num2str(i-1)],'WriteVariableNames',0);
end

ROIinfo = array2table(ROIinfo);
writetable(ROIinfo,[stk,'_Puncti.xlsx'],'Sheet',['Parameters'],'WriteVariableNames',0);

msgbox('Done')
