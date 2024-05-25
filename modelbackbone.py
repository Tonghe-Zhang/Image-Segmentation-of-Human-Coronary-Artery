import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.models as models
class conv_block(nn.Module):
    """
    Input shape:   (N, ch_in, H, W)
    Output shape:  (N, ch_out, H, W)
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
        Input shape:        Output shape:
      (N, ch_in, H, W)-> (N, ch_out, 2H, 2W)
    """
    def __init__(self, ch_in:int, ch_out:int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        """
        Upsample(): N, C, H, W -> N, C, H*scale_factor, W*scale_factor
        """
    def forward(self, x):
        x = self.up(x)
        return x

class UNetDecoderLayer(nn.Module):
    """"
    Input shape:   x_link=(N, ch_in_1, Hx2, Wx2)     decoded=(N, ch_in_2, H, W)
    Output shape:  (N, ch_out, H, W)
    """
    def __init__(self, ch_in:int, ch_cat:int, ch_out:int, prt_progress=False):
        super(UNetDecoderLayer, self).__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.prt_progress=prt_progress

        self.UpConv = up_conv(ch_in, ch_out)#
        self.Conv = conv_block(ch_cat+ch_out, ch_out)
    def forward(self,x_link,decoded):
        """"
        decoded:        N, Cin,     H, W
        >up_conv
        decoded_up:     N, Cout,    H*2, W*2
        x_link:         N, Ccat,      Hx=2H, Wx=2W
        >cat
        decoded_cat:    N, Ccat+Cout  2H, 2W 
        >conv1
        decoded:        N, Cout,    2H, 2W         [constraint:  Cin=Ccat+Cout]
        """
        if self.prt_progress:
            print(f"ch_in={self.ch_in}, ch_out={self.ch_out}")
            print(f"x_link={x_link.shape  if x_link is not None else None}, d={decoded.shape}")

        # up convolution, (N, ch_in, H, W)-> (N, ch_out, 2H, 2W)
        decoded_up=self.UpConv(decoded)

        if self.prt_progress:
            print(f"after up, decoded={decoded.shape}, decoded_up={decoded_up.shape}")

        # concatenate with encoder feature (except for the last layer),      (N, ch_out, 2H, 2W) -> (N, ch_out+Ccat, 2H, 2W)  
        decoded_cat=torch.cat([x_link,decoded_up],dim=1) if x_link is not None else decoded_up

        if self.prt_progress:
            print(f"after cat, decoded_cat={decoded_cat.shape}")

        # second convlution, (N, ch_out+Ccat, 2H, 2W) -> (N, Cout, 2H, 2W)
        decoded=self.Conv(decoded_cat)

        return decoded

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder_feat_ch:list,
                 decoder_feat_ch:list,
                 prt_progress=False
                 ):
        super(UNetDecoder, self).__init__()
        self.output_W=512
        self.output_H=512
        self.prt_progress=prt_progress

        # determine the channel of each layer
        self.num_layers=len(decoder_feat_ch)
        # output channels
        self.layer_Couts=decoder_feat_ch                        #output channel dimension of each decoder layer.
        assert decoder_feat_ch[-1]==2
        self.output_ch=decoder_feat_ch[-1]                      #output segmentation image channel dimension, which is 2 by default (grey figure)
        # input channels
        self.layer_Cins=[_ for _ in range(self.num_layers)]     #input channel dimension of each decoder layer.
        self.layer_Cins[0]=encoder_feat_ch[0]
        self.layer_Cins[1:]=self.layer_Couts[0:-1]
        # concatnate channels
        self.layer_Ccats=encoder_feat_ch[1:]
        self.layer_Ccats.append(None)                           # the last layer is full conv, does not need concatenation. 
        
        # create decoder stack.
        self.decoder_layers=nn.ModuleList()
        # From the deepest layer to the second highest layer, we use concatenation. 
        for ch_in, ch_cat, ch_out in zip(self.layer_Cins[:-1], self.layer_Ccats[:-1], self.layer_Couts[:-1]):
            self.decoder_layers.append(UNetDecoderLayer(ch_in=ch_in, ch_cat=ch_cat, ch_out=ch_out,prt_progress=self.prt_progress))
        
        # The last layer outputs hidden features to a grey figure distribution(prediction masks), and it does not need concatnation. 
        #output_conv=nn.Conv2d(in_channels=decoder_feat_ch[-2], out_channels=self.output_ch, kernel_size=1, stride=1, padding=0)
        #self.decoder_layers.append(output_conv)
        self.decoder_layers.append(UNetDecoderLayer(ch_in=decoder_feat_ch[-2], ch_cat=0, ch_out=self.output_ch,prt_progress=self.prt_progress))

        if self.prt_progress:
            print(f"Decoder layer channels: Cins={self.layer_Cins}, Ccats={self.layer_Ccats}, Couts={self.layer_Couts}")
            print(f"Decoder layers:{self.decoder_layers}")
            #exit()

    def forward(self,encoded_features):
        # ... x5,x4,x3,x2,x1 = encoded_features
        decoded_features=[ _ for _ in range(self.num_layers) ]

        # check whether the spatil sizes of encoded featues decrease at the rate of 2^{-n}
        if encoded_features[0].shape[2:] != torch.Size([self.output_H//2**(self.num_layers), self.output_W//2**(self.num_layers)]):
            raise ValueError(f"Expected encoder output of shape [N, C, {self.output_H//2**(self.num_layers)}, {self.output_W//2**(self.num_layers)}], but got {encoded_features[0].shape} !")

        # ...d5, d4, d3, d2, d1 =decoded_features
        d=encoded_features[0]
        decoded_features[0]=self.decoder_layers[0](x_link=encoded_features[1],decoded=encoded_features[0])
        for feat in range(1,self.num_layers-1):
            d=decoded_features[feat-1]
            x=encoded_features[feat+1]
            d=self.decoder_layers[feat](x_link=x,decoded=d)
            decoded_features[feat]=d
        d=self.decoder_layers[-1](x_link=None,decoded=decoded_features[-2])    # in the last layer we only do upsampling and convolution, but do not need to concatenate any encoder feature.
        decoded_features[-1]=d

        if self.prt_progress:
            for i, feat in enumerate(encoded_features):
                print(f"encoded{i} ={feat.shape}")
            for i, feat in enumerate(decoded_features):
                print(f"decoded{i} ={feat.shape}")

        #check output image must be the same as the labels (N, 2, 512, 512) in our case. 
        if d.shape[1:] != torch.Size([2,self.output_H,self.output_W]):
            raise ValueError(f"Expected decoder output of shape [N,2,{self.output_H},{self.output_W}], but got {d.shape} !")
        return d

class UNetBackBoneEncoder(nn.Module):
    def __init__(self, backbone_name='resnet50',
                 extraction_layer_names=['relu', 'layer1', 'layer2', 'layer3','layer4'],
                 freeze_backbone=True,
                 prt_prgress=False):
        """
        Define the backbone model. it should be a pretrained model like resnet family. 
        We also need to define the layers from which we extract features. 
        We will pass the data through the entire pretrained model and only extract features from sepcific layers.
        To save compute we will terminate the single pass process when the data reached the last layer from which feature are extracted.
        """
        super(UNetBackBoneEncoder, self).__init__()
        self.extraction_layer_names=extraction_layer_names
        self.num_extraction_layers=len(self.extraction_layer_names)
        self.prt_progress=prt_prgress
        
        # assert self.num_extraction_layers == 5
        
        if backbone_name=='resnet50':
            self.backbone=models.resnet50(pretrained=True)
            # modify the first input layer to allow for grey image input. 
            self.backbone.conv1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            raise NotImplementedError(f"We have only implemented backboned encoder for resnet50, but we got {backbone_name} instead !")

        if freeze_backbone:
            #Todo: better not fix the first input layer, as it is not learnt.
            for param in self.backbone.parameters():
                param.requires_grad=False
            for param in self.backbone.conv1.parameters():
                param.requires_grad=True

        # automatically discover the encoded feature' channels
        x=torch.randn(2,1,2,2)
        encoded_ch=[]
        for name, child in self.backbone.named_children():
            x=child(x)
            if name in self.extraction_layer_names:
                encoded_ch.append(x.shape[1])
                if name==self.extraction_layer_names[-1]:
                    break
        encoded_ch.reverse()
        self.encoded_ch=encoded_ch

        if prt_prgress:
            print(f"encoded_ch={self.encoded_ch}")


    def forward(self,x):
        """
        We will pass the input data through all the layers of the pretrained model.
        And we will only extract features from specific layers, which will be added to a disctionary "encoded_features". 

        Input: data of shape N, 1, 512, 512
        Output: a list of tensors, which are the encoded features in a top-to-down order. 
        """
        encoded_features=[_ for _ in range(self.num_extraction_layers)]
        i=0
        for name, child in self.backbone.named_children():
            # single pass through all the layers...
            x=child(x)
            if name in self.extraction_layer_names:
                encoded_features[i]=x
                i+=1
                if name==self.extraction_layer_names[-1]:
                    break
        
        if self.prt_progress:
            print(f"there are {len(encoded_features)} layers of feature being extracted as encoded_feature. ")
            for name, feat in zip(self.extraction_layer_names, encoded_features):
                print(f"after name of layer={name}, output is of shape {feat.shape}")
        
        # return the features in a reversed order: x5, x4, x3, x2, x1
        encoded_features.reverse()
        return encoded_features


class BackBonedUNet(nn.Module):
    def __init__(self,
                 backbone_name='resnet50',
                 extraction_layer_names=['relu', 'layer1', 'layer2', 'layer3','layer4'],
                 freeze_backbone=True,
                 decoder_feat_ch=[512, 256, 128, 64, 2],
                 prt_prgress=False):
        """
        Backboned UNet. 
        We use the pretrained model specified by ``backbone_name'' to extract features from raw input grey figure of shape N, 1, 512, 512. 
        We have adjusted the pretrained model's input layer to accomodate grey image input. We will freeze the pretrained model' weighs (except for the 
        newly defined first layer) when ``freeze_backbone'' is False. Otherwise while training the BackBonedUNet we will also finetune the pretrained model. 
        
        The Unet will adjust its hidden layer's shapes according to the encoder's extracted features and user-defined decoder feature' dimensions. 
        The only constraint is that the extracted features from the encoder should reduce their heights and widths by half, and there is no constraint on 
        the decoder features' channels. 

        During forward() call, we pass the data through the entire pretrained model and only extract features from sepcific layers indiced by ``extraction_layer_names''. 
        Then we feed these feature(five layers in default) to our Unet decoder, whose features' channels are predifined by the user according to ``decoder_feat_ch''.        
        """
        super(BackBonedUNet, self).__init__()

        self.prt_progress=prt_prgress
        # build encoder stack
        if self.prt_progress:
            print(f"Build encoder stack:")
        self.encoder=UNetBackBoneEncoder(backbone_name=backbone_name,
                                         extraction_layer_names= extraction_layer_names,
                                         prt_prgress=self.prt_progress)
        self.encode_ch=self.encoder.encoded_ch

        # build decoder stack
        if self.prt_progress:
            print(f"Build decoder stack:")
        self.decoder=UNetDecoder(encoder_feat_ch=self.encode_ch,
                        decoder_feat_ch=decoder_feat_ch,
                        prt_progress=self.prt_progress)
    
    def forward(self,x):
        if self.prt_progress:
            print(f"Forward pass through encoder stack:")
        encoded_features=self.encoder(x)
        if self.prt_progress:
            print(f"Forward pass through decoder stack:")
        predicted_masks=self.decoder(encoded_features)
        if self.prt_progress:
            print(f"Return predicted segmentation mask of shape{predicted_masks.shape}")
        return predicted_masks

