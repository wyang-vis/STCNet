
from .net_util import *


class Modality_Spatial_Collaboration(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Modality_Spatial_Collaboration, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.CMC=Cross_Modal_Calibration(dim, num_heads, bias)
        self.CMA=Cross_Modal_Aggregation(dim)
        self.project_out = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        #################################CMC###################
        F_dm=image-event
        CMC_img=self.CMC(self.norm1_image(F_dm), self.norm1_event(image))
        CMC_event=self.CMC(self.norm1_image(F_dm), self.norm1_event(event))
        #################################CMA###################
        fuse_img_to_event=self.CMA(CMC_img,event)
        fuse_event_to_img=self.CMA(CMC_event,image)
        fuse_cat = torch.cat((fuse_img_to_event, fuse_event_to_img), 1)
        fused=self.project_out(fuse_cat)
        return fused





class Modality_Temporal_Collaboration(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Modality_Temporal_Collaboration, self).__init__()
        self.num_heads=num_heads
        self.norm1_image1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_event1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_image2 = LayerNorm(dim, LayerNorm_type)
        self.norm1_event2 = LayerNorm(dim, LayerNorm_type)
        self.Cross_attn_img = Spatio_Temporal_Mutual_Attention(dim, num_heads, bias)
        self.Cross_attn_event = Spatio_Temporal_Mutual_Attention(dim, num_heads, bias)
        self.MCA= Multimodal_Coordinate_Attention(dim)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)



    def forward(self, image1,image, image2, event1,event,event2, fusion):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        attn_img1,v_img1=self.Cross_attn_img(self.norm1_image1(image1), self.norm1_image2(image))
        attn_event1,v_event1=self.Cross_attn_event(self.norm1_event1(event1), self.norm1_event2(event))
        Mul_attn_1=attn_img1*attn_event1
        Mul_out_1 = (Mul_attn_1 @ v_img1)+(Mul_attn_1 @ v_event1)
        Mul_out_1 = rearrange(Mul_out_1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        Mul_out_1 = self.project_out1(Mul_out_1)


        attn_img2,v_img2=self.Cross_attn_img(self.norm1_image1(image2), self.norm1_image2(image))
        attn_event2,v_event2=self.Cross_attn_event(self.norm1_event1(event2), self.norm1_event2(event))
        Mul_attn_2=attn_img2*attn_event2
        Mul_out_2 = (Mul_attn_2 @ v_img2)+(Mul_attn_2 @ v_event2)
        Mul_out_2 = rearrange(Mul_out_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        Mul_out_2 = self.project_out2(Mul_out_2)

        ST_fusion_1=self.MCA(fusion,Mul_out_1)
        ST_fusion_2=self.MCA(fusion,Mul_out_2)
        ST_fusion=self.project_out(torch.cat([ST_fusion_1,ST_fusion_2], dim=1))

        return ST_fusion


class Encoder(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels):
        super(Encoder, self).__init__()
        # self.inplanes = 32
        self.num_heads=4
        ######encoder
        self.head = shallow_cell(inChannels)
        self.down1 = EN_Block(64, 128)  # 128
        self.down2 = EN_Block(128, 256)  # 64


    def forward(self, input):

        s0 = self.head(input)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = [s0, s1, s2]
        return x


class Decoder(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, outChannels):
        super(Decoder, self).__init__()
        ######Decoder
        # self.up1 = DE_Block(512, 256)
        # self.up2 = DE_Block(256, 128)
        self.up3 = DE_Block(256, 128)
        self.up4 = DE_Block(128, 64)

    def forward(self, input,skip):
        x=input
        x = self.up3(x, skip[1])
        x = self.up4(x, skip[0])
        return x


class Restoration(nn.Module):
    """Modified version of Unet from SuperSloMo.
    """

    def __init__(self, inChannels_img, inChannels_event,outChannels, args,ends_with_relu=False):
        super(Restoration, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.num_ff = args.future_frames
        self.num_fb = args.past_frames
        self.num_heads=4
        ######encoder部分
        self.encoder_img=Encoder(inChannels_img)
        self.encoder_event=Encoder(inChannels_event)
        self.decoder = Decoder(outChannels)
        ######fusion部分
        self.MSC = Modality_Spatial_Collaboration(256, num_heads=self.num_heads,ffn_expansion_factor=4, bias=False,
                                                                                   LayerNorm_type='WithBias')

        self.MTC = Modality_Temporal_Collaboration(256, num_heads=self.num_heads,ffn_expansion_factor=4, bias=False,
                                                                                   LayerNorm_type='WithBias')

        self.conv = nn.Conv2d(64, outChannels, 3, stride=1, padding=1)


    def forward(self, input_img, input_event):
        #### Multi-scale feature extraction

        batch_size, frames, channels, height, width = input_img.shape
        en_out_img = []
        en_out_event = []
        Fuse_t = []
        for t in range(frames):
            out_img = self.encoder_img(input_img[:, t, :, :, :])
            out_event = self.encoder_event(input_event[:, t, :, :, :])
            en_out_img.append(out_img)
            en_out_event.append(out_event)
            F_t=self.MSC(out_img[-1],out_event[-1])
            Fuse_t.append(F_t)

        ST_Fusion=self.MTC(en_out_img[0][-1],en_out_img[1][-1],en_out_img[2][-1],
                                                   en_out_event[0][-1],en_out_event[1][-1],en_out_event[2][-1],
                                                   Fuse_t[1])
        out=self.decoder(ST_Fusion,en_out_img[1])
        out = self.conv(out)
        out=out+input_img[:, 1, :, :, :]
        return out
