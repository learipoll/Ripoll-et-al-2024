% Written by Léa Ripoll
% Quantifies the background subtracted fluorescence at segmented endosomes

broot = cd;
parentfile = dir;
 
for i = 3:size(parentfile,1)
 
   if parentfile(i).isdir  %if directory, get in it
   cd([broot,'/',parentfile(i).name])
  	 
   final = {'File name','Fluo cell','Fluo endosomes','Enrichment endosomes'};

 
 % Get the list of files in the folder
    listfile = dir;
   
    for j = 3:size(listfile,1)
        
    if listfile(j).isdir
        cd([broot,'/',parentfile(i).name,'/',listfile(j).name])
        
        listcell = dir;
                
        for k = 4:size(listcell,1) % browsing through files in image folder
            
            %check if file is an image
            filename = listcell(k,1).name;
            
                if contains(filename,'green.tif')
                    %initialize 3D matrix
                    M=[];
                    
                    %get stack info
                    info = imfinfo(filename);  %if know directly the name of file, replace stk by name
                    %get number of frame
                    num_images = numel(info);
                    %size(info,1) seems to work too
                    %read every frame with a loop
                    
                    for l = 1:num_images
                    A = imread(filename, l);
                    M=cat(3,M,A);     
                    
                    end
                        
                    M = double(M); %convert image to double class
                    Matc1 = max(M,[],3); 
                    
                end
                
                if contains(filename,'red.tif')
                    %initialize 3D matrix
                    M=[];
                    
                    %get stack info
                    info = imfinfo(filename);  %if know directly the name of file, replace stk by name
                    %get number of frame
                    num_images = numel(info);
                    %size(info,1) seems to work too
                    %read every frame with a loop
                    
                    for m = 1:num_images
                    A = imread(filename,m);
                    M=cat(3,M,A);     
                    
                    end
                    
                    M = double(M); %convert image to double class
                    Matc2 = max(M,[],3);
                    
                end
                
        end
        
        %draw polygon on c1 to contour the cell
        image(Matc1,'cdatamapping','scaled');
        colormap(gray)
        axis image
        caxis([600 6000])
        title('Draw the contour of the cell')
        polygonc1 = impoly;
        cellMask = createMask(polygonc1);
        cellMask = double(cellMask);
        
        %draw crop on c1 to measure the background
        image(Matc1,'cdatamapping','scaled')
        colormap(gray)
        axis image
        caxis([600 6000])
        title('Draw the background region')
        cropc1 = imcrop;
        
        %measure background on c1 and c2
        backc1 = sum(cropc1,'all')/(size(cropc1,1)*size(cropc1,2));
        
        %threshold c2 in polygon to make a mask
        endoMask = cellMask .* Matc2;
        endoMask1 = (endoMask > 5000); %make threshold at 5,000
        endoMask1 = double(endoMask1);
                
        %measure fluo c1 in the endosome mask
        c1endo1 = endoMask1 .* Matc1;
        Fluo_endo1 = sum(c1endo1,'all')/sum(endoMask1,'all');
        Fluo_endo_back1 = Fluo_endo1 - backc1;
        
        %measure fluo c1 in the total cell
        c1cell = cellMask .* Matc1;
        Fluo_cell = sum(c1cell,'all')/sum(cellMask,'all');
        Fluo_cell_back = Fluo_cell - backc1;
        
        %measure ratio endo/cell
        Enrich_endo1 = Fluo_endo_back1/Fluo_cell_back;
        display(Enrich_endo1)
        cd ..
            
        %save the maximum projections
        Maxc1 = uint16(Matc1);
        Maxc2 = uint16(Matc2);
        imwrite(Maxc1,[listfile(j).name,'_max_proj_','green','.tif'],'Compression','none','WriteMode','append')
        imwrite(Maxc2,[listfile(j).name,'_max_proj_','red','.tif'],'Compression','none','WriteMode','append')
        
        %save the masks
        CellMask = uint16(cellMask);
        EndoMask1 = uint16(endoMask1);

        imwrite(CellMask,[listfile(j).name,'_cell_mask','.tif'],'Compression','none','WriteMode','append')
        imwrite(EndoMask1,[listfile(j).name,'_endo_mask','.tif'],'Compression','none','WriteMode','append')    
        final = cat(1,final,{listfile(j).name,Fluo_cell_back,Fluo_endo_back1,Enrich_endo1});
        
    end
    
    end
    
    cd ..
        
    final = array2table(final);
    writetable(final,[parentfile(i).name,'_Fluo_endosomes.xls']); 
    
   end       

end

close(gcf)

msgbox('Well done Lea')

