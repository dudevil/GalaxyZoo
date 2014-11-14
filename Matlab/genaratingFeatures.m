imagesList = dir('.\images_training_rev1');

rng = 0:26;
nbins = 10;
histRange = rng*nbins;
bitscount = [];
    
A  = zeros(length(imagesList),length(rng)*3 + 3);

basePath = cd();
for i=1:length(imagesList)
    
    fullImageName = strcat(basePath ,'\images_training_rev1\', imagesList(i).name);
    try
        img = imread(fullImageName);  
    catch err
        disp (err.message);
        continue
    end
    imgR = img(:,:,1);
    imgG = img(:,:,2);
    imgB = img(:,:,3);
     
    bitscountR = histc(imgR(:),histRange);
    bitscountG = histc(imgG(:),histRange);
    bitscountB = histc(imgB(:),histRange);

    entropyR = entropy(imgR);
    entropyG = entropy(imgG);
    entropyB = entropy(imgB);
    
    A(i,:) = cat(1,bitscountR,bitscountG,bitscountB, entropyR, entropyG, entropyB);  
end

csvwrite('data.csv',A);