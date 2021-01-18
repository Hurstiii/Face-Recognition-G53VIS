function C = croptoface(D1)
%CROPTOFACE Summary of this function goes here
%   Detailed explanation goes here
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');

% apply the face detector
BB = step(faceDetector, D1);

if isempty(BB) % if it didn't detect a face
    C = double(D1);
else
    if (size(BB,1) > 1)
        [M, I] = max(BB(:,3));
        C = double(imcrop(D1, BB(I, :)));
    else
        C = double(imcrop(D1, BB));
    end
end
end

