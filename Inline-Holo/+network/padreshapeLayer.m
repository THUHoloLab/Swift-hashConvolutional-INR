classdef padreshapeLayer < nnet.layer.Layer & ...
                           nnet.layer.Formattable & ...
                           nnet.layer.Acceleratable

    properties 
        imgSize;
    end

    methods 
        function self = padreshapeLayer(args)
            arguments
                args.imgSize = [256,256];
                args.name = "Pad and Reshape"
            end

            self.Name = args.name;
            self.imgSize = args.imgSize;
        end

        function Y = predict(self,X) 
            dim = finddim(X,"B");
            fea = finddim(X,"C");
            
            X = stripdims(X);
            if size(X,dim) < prod(self.imgSize)
                Y = dlarray(gpuArray.zeros(size(X,fea),...
                                           prod(self.imgSize),'single'));
                Y(:,1:size(X,dim)) = X;
                disp('this case');
            else
                Y = X;
            end
            Y = reshape(Y', self.imgSize(1),...
                            self.imgSize(2),...
                            size(Y,fea));

            Y = dlarray(Y,"SSCB");
        end
    end
end
