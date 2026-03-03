classdef Resize_ImageLayer < nnet.layer.Layer & ...
                             nnet.layer.Formattable

    properties 
        imgSize;
    end


    methods 
        function self = Hash_EncodeingLayer(args)
            arguments
                args.imgSize = [256,256];
                args.name = "Reshape-image"
            end

            self.Name = args.name;
            self.imgSize = args.imgSize;
        end

        function Y = predict(self,X) 
            dim = finddim(X,"B");

            if size(X,dim) > 1
                Y = stripdims(X);
                Y = reshape(Y,[self.imgSize(1),...
                               self.imgSize(2),...
                               size(X,finddim(X,"C"))]);
            end
            Y = dlarray(Y,"SSCB");
        end
    end
end

