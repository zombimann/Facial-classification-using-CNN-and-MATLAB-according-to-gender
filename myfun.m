
function y = myfun (img, inputSize) % , thresh)
% y = imbinarize(y, thresh);
% y = imbinarize(y);
% faceDetector = vision.CascadeObjectDetector;
% bboxes = step(faceDetector, img);
% try
%     face = img(bboxes(1,2):bboxes(1,2)+bboxes(1,4),bboxes(1,1):bboxes(1,1)+bboxes(1,3));
%     y = imresize(face, inputSize);
% catch
    y = imresize(img, inputSize);
    %y = imadjust(y,[.3 .4 0; .7 .8 1],[]);
    y = imsharpen(y,'Radius',1,'Amount',2);
% end

end

