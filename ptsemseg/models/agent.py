import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import pretrainedmodels
import copy

from ptsemseg.models.utils import conv2DBatchNormRelu, deconv2DBatchNormRelu, Sparsemax
import random
from torch.distributions.categorical import Categorical

from ptsemseg.models.backbone import n_segnet_encoder, resnet_encoder, n_segnet_decoder, simple_decoder, FCN_decoder
import numpy as np


def get_encoder(name):
    try:
        return {
            "n_segnet_encoder": n_segnet_encoder,
            "resnet_encoder": resnet_encoder,
        }[name]
    except:
        raise ("Encoder {} not available".format(name))


def get_decoder(name):
    try:
        return {
            "n_segnet_decoder": n_segnet_decoder,
            "simple_decoder": simple_decoder,
            "FCN_decoder": FCN_decoder

        }[name]
    except:
        raise ("Decoder {} not available".format(name))


### ============= Modules ============= ###
class img_encoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, feat_channel=512, feat_squeezer=-1,
                 enc_backbone='n_segnet_encoder'):
        super(img_encoder, self).__init__()
        feat_chn = 256

        self.feature_backbone = get_encoder(enc_backbone)(n_classes=n_classes, in_channels=in_channels)
        self.feat_squeezer = feat_squeezer

        # squeeze the feature map size 
        if feat_squeezer == 2:  # resolution/2
            self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=2, padding=1)
        elif feat_squeezer == 4:  # resolution/4
            self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=4, padding=1)
        else:
            self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.feature_backbone(inputs)
        outputs = self.squeezer(outputs)

        return outputs


class img_decoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=512, agent_num=5, feat_squeezer=-1, dec_backbone='n_segnet_decoder'):
        super(img_decoder, self).__init__()

        self.feat_squeezer = feat_squeezer
        if feat_squeezer == 2:  # resolution/2
            self.desqueezer = deconv2DBatchNormRelu(in_channels, in_channels, k_size=3, stride=2, padding=1,
                                                    output_padding=1)
            self.output_decoder = get_decoder(dec_backbone)(n_classes=n_classes, in_channels=in_channels)

        elif feat_squeezer == 4:  # resolution/4
            self.desqueezer1 = deconv2DBatchNormRelu(in_channels, 512, k_size=3, stride=2, padding=1, output_padding=1)
            self.desqueezer2 = deconv2DBatchNormRelu(512, 512, k_size=3, stride=2, padding=1, output_padding=1)
            self.output_decoder = get_decoder(dec_backbone)(n_classes=n_classes, in_channels=512)
        else:
            self.output_decoder = get_decoder(dec_backbone)(n_classes=n_classes, in_channels=in_channels)

    def forward(self, inputs):
        if self.feat_squeezer == 2:  # resolution/2
            inputs = self.desqueezer(inputs)

        elif self.feat_squeezer == 4:  # resolution/4
            inputs = self.desqueezer1(inputs)
            inputs = self.desqueezer2(inputs)

        outputs = self.output_decoder(inputs)
        return outputs


class msg_generator(nn.Module):
    def __init__(self, in_channels=512, message_size=32):
        super(msg_generator, self).__init__()
        self.in_channels = in_channels

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(self.in_channels, 256, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(256, 128, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(128, 64, k_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(64, message_size, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs


class policy_net4(nn.Module):
    def __init__(self, n_classes=21, in_channels=512, input_feat_sz=32, enc_backbone='n_segnet_encoder'):
        super(policy_net4, self).__init__()
        self.in_channels = in_channels

        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)

        self.img_encoder = img_encoder(n_classes=n_classes, in_channels=self.in_channels, enc_backbone=enc_backbone)

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        outputs1 = self.img_encoder(features_map)

        outputs = self.conv1(outputs1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs


class km_generator(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32):
        super(km_generator, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #            

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs


class linear(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32):
        super(linear, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)

        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)
        )

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs


class conv(nn.Module):
    def __init__(self, out_size=128):
        super(conv, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.conv = conv2DBatchNormRelu(256, out_size, k_size=1, stride=1, padding=1)

    def forward(self, features_map):
        outputs = self.conv(features_map)
        return outputs



# <------ Attention ------> #
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, sparse=True):
        attn_orig = torch.bmm(k, q.transpose(2, 1))
        attn_orig = attn_orig / self.temperature
        if sparse:
            attn_orig = self.sparsemax(attn_orig)  
        else:
            attn_orig = self.softmax(attn_orig)  
        attn = torch.unsqueeze(torch.unsqueeze(attn_orig, 3), 4)  
        output = attn * v  # (batch,4,channel,size,size)
        output = output.sum(1)  # (batch,1,channel,size,size)
        return output, attn_orig.transpose(2, 1)

class AdditiveAttentin(nn.Module):
    def __init__(self):
        super().__init__()
        # self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)
        self.sparsemax = Sparsemax(dim=1)
        self.linear_feat = nn.Linear(128, 128)
        self.linear_context = nn.Linear(128, 128)
        self.linear_out = nn.Linear(128, 1)

    def forward(self, q, k, v, sparse=True):
        # q (batch,1,128)
        # k (batch,4,128)
        # v (batch,4,channel,size,size)
        temp1 = self.linear_feat(k)  # (batch,4,128)
        temp2 = self.linear_context(q)  # (batch,1,128)
        attn_orig = self.linear_out(temp1 + temp2)  # (batch,4,1)
        if sparse:
            attn_orig = self.sparsemax(attn_orig)  # (batch,4,1)
        else:
            attn_orig = self.softmax(attn_orig)  # (batch,4,1)
        attn = torch.unsqueeze(torch.unsqueeze(attn_orig, 3), 4)  # (batch,4,1,1,1)
        output = attn * v
        output = output.sum(1)  # (batch,1,channel,size,size)
        return output, attn_orig.transpose(2, 1)

# MIMO (non warp)
class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,6,32)
        # k (batch,6,1024)
        # v (batch,6,channel,size,size) [2, 6, 512, 16, 16]

        query = self.linear(qu)  # [2, 6, 1024]

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)



        # generate the
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch, 6, 6)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,6,6)

        attn_shape = attn_orig_softmax.shape # (2,6,6)

        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]

        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1) #[2, 6, 6, 1, 1, 1]

        v_exp = torch.unsqueeze(v, 2) #[2, 6, 1, 512, 16, 16])

        v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1) #[2, 6, 6, 512, 16, 16])

        output = attn_orig_softmax_exp * v_exp  # (batch,4,channel,size,size) - [2, 6, 6, 512, 16, 16]

        output_sum = output.sum(1)  # (batch,1,channel,size,size) - [2, 1, 6, 512, 16, 16]

        return output_sum, attn_orig_softmax

# MIMO always com
class MIMOWhoGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,5,key_size)


        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch,5,5)  column: differnt keys and the same query


        # remove the diagonal and softmax
        del_diag_att_orig = []
        for bi in range(attn_orig.shape[0]):
            up = torch.triu(attn_orig[bi],diagonal=1,out=None)[:-1,] 
            dow = torch.tril(attn_orig[bi],diagonal=-1,out=None)[1:,] 
            del_diag_att_orig_per_sample = torch.unsqueeze((up+dow),dim=0)
            del_diag_att_orig.append(del_diag_att_orig_per_sample)
        del_diag_att_orig = torch.cat(tuple(del_diag_att_orig), dim=0)

        attn_orig_softmax = self.softmax(del_diag_att_orig)  # (batch,5,5)

        append_att_orig = []
        for bi in range(attn_orig_softmax.shape[0]):
            up = torch.triu(attn_orig_softmax[bi],diagonal=1,out=None)
            up_ext = torch.cat((up, torch.zeros((1, up.shape[1])).cuda()))
            dow = torch.tril(attn_orig_softmax[bi],diagonal=0,out=None)
            dow_ext = torch.cat((torch.zeros((1, dow.shape[1])).cuda(), dow))

            append_att_orig_per_sample = torch.unsqueeze((up_ext + dow_ext),dim=0)
            append_att_orig.append(append_att_orig_per_sample)
        append_att_orig = torch.cat(tuple(append_att_orig), dim=0)



        attn_shape = append_att_orig.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = append_att_orig.view(bats, key_num, query_num, 1, 1, 1)

        v_exp = torch.unsqueeze(v, 2)
        v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (batch,4,channel,size,size)
        output_sum = output.sum(1)  # (batch,1,channel,size,size)

        return output_sum, append_att_orig

class GeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, q, k, v, sparse=True):
        # q (batch,1,128)
        # k (batch,4,128)
        # v (batch,4,channel*size*size)
        query = self.linear(q)  # (batch,1,key_size)
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch,4,1)
        if sparse:
            attn_orig = self.sparsemax(attn_orig)  # (batch,4,1)
        else:
            attn_orig = self.softmax(attn_orig)  # (batch,4,1)
        attn = torch.unsqueeze(torch.unsqueeze(attn_orig, 3), 4)  # (batch,4,1,1,1)
        output = attn * v  # (batch,4,channel,size,size)
        output = output.sum(1)  # (batch,1,channel,size,size)
        return output, attn_orig.transpose(2, 1)

    # ============= Single normal and Single degarded  ============= #


# =======================  Model =========================


# Randomly selection baseline and Concatenation of all observations
class All_agents(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, feat_channel=512, aux_agent_num=4, shuffle_flag=False,
                 enc_backbone='n_segnet_encoder', dec_backbone='n_segnet_decoder', feat_squeezer=-1):
        super(All_agents, self).__init__()

        self.agent_num = aux_agent_num  
        self.in_channels = in_channels
        self.shuffle_flag = shuffle_flag

        # Encoder
        self.encoder1 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                    feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
        self.encoder2 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                    feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
        self.encoder3 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                    feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
        self.encoder4 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                    feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
        self.encoder5 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                    feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)

        # Decoder for interested agent
        if self.shuffle_flag == 'selection': # random selection
            self.decoder = img_decoder(n_classes=n_classes, in_channels=feat_channel * 2 ,
                                    feat_squeezer=feat_squeezer, dec_backbone=dec_backbone)
        else: # catall
            self.decoder = img_decoder(n_classes=n_classes, in_channels=feat_channel * self.agent_num,
                                    feat_squeezer=feat_squeezer, dec_backbone=dec_backbone)

    def divide_inputs(self, inputs):
        '''
        Divide the input into a list of several images
        '''
        input_list = []
        divide_num = 5
        for i in range(divide_num):
            input_list.append(inputs[:, 3 * i:3 * i + 3, :, :])

        return input_list

    def forward(self, inputs):
        # agent_num = 5

        input_list = self.divide_inputs(inputs)
        feat_map1 = self.encoder1(input_list[0])
        feat_map2 = self.encoder2(input_list[1])
        feat_map3 = self.encoder3(input_list[2])
        feat_map4 = self.encoder4(input_list[3])
        feat_map5 = self.encoder5(input_list[4])

        if self.shuffle_flag == 'selection':  # use randomly picked feature and only specific numbers
            aux_view_feats = [feat_map1,feat_map2, feat_map3, feat_map4, feat_map5]
            aux_id = random.randint(0, 4)
            aux_view_feats = torch.unsqueeze(aux_view_feats[aux_id], 0)
            feat_map_list = (feat_map1,) + tuple(aux_view_feats)
            argmax_action = torch.ones(feat_map1.shape[0], dtype=torch.long)*aux_id

        elif self.shuffle_flag == 'fixed2':
            feat_map_list = (feat_map1, feat_map2)

        else:
            feat_map_list = (feat_map1, feat_map2, feat_map3, feat_map4, feat_map5)

        # combine the feat maps
        concat_featmaps = torch.cat(feat_map_list, 1)
        pred = self.decoder(concat_featmaps)

        if self.shuffle_flag == 'selection':  # use randomly picked feature and only specific numbers
            return pred, argmax_action.cuda()
        else:
            return pred

# When2Com - Multiple inputs $ outputs (no warping)
class MIMOcom(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, agent_num=5, shuffle_flag=False, image_size=512,
                 shared_img_encoder=False, key_size=128, query_size=128, enc_backbone='n_segnet_encoder',
                 dec_backbone='n_segnet_decoder'):
        super(MIMOcom, self).__init__()

        self.agent_num = agent_num
        self.in_channels = in_channels
        self.shuffle_flag = shuffle_flag
        self.feature_map_channel = 512
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.sparse = sparse


        print('When2com') # our model: detach the learning of values and keys
        self.u_encoder = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                         feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)

        self.key_net = km_generator(out_size=self.key_size, input_feat_sz=image_size / 32)
        self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)

        # # Message generator
        self.query_key_net = policy_net4(n_classes=n_classes, in_channels=in_channels, enc_backbone=enc_backbone)
        if self.has_query:
            self.query_net = km_generator(out_size=self.query_size, input_feat_sz=image_size / 32)

        # Segmentation decoder
        self.decoder = img_decoder(n_classes=n_classes, in_channels=self.feature_map_channel,
                                   feat_squeezer=feat_squeezer, dec_backbone=dec_backbone)


        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        if self.shared_img_encoder == 'unified':
            self.img_net_paras = list(self.u_encoder.parameters()) + list(self.decoder.parameters())



        self.policy_net_paras = list(self.query_key_net.parameters()) + list(
            self.key_net.parameters()) + self.attention_paras
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(self.query_net.parameters())

        self.all_paras = self.img_net_paras + self.policy_net_paras


    def shape_feat_map(self, feat, size):
        return torch.unsqueeze(feat.view(-1, size[1] * size[2] * size[3]), 1)

    def argmax_select(self, val_mat, prob_action):
        # v(batch, query_num, channel, size, size)
        cls_num = prob_action.shape[1]
        coef_argmax = F.one_hot(prob_action.max(dim=1)[1],  num_classes=cls_num).type(torch.cuda.FloatTensor)
        coef_argmax = coef_argmax.transpose(1, 2)
        attn_shape = coef_argmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]

        coef_argmax_exp = coef_argmax.view(bats, key_num, query_num, 1, 1, 1)
        v_exp = torch.unsqueeze(val_mat, 2)
        v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)
        output = coef_argmax_exp * v_exp  # (batch,4,channel,size,size)
        feat_argmax = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = copy.deepcopy(coef_argmax)
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0])
        return feat_argmax, coef_argmax, num_connect

    def activated_select(self, val_mat, prob_action, thres=0.2):

        coef_act = torch.mul(prob_action, (prob_action > thres).float())
        attn_shape = coef_act.shape   #[2, 6, 6]

        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1)     #[2, 6, 6, 1, 1, 1]

        v_exp = torch.unsqueeze(val_mat, 2)     #[2, 6, 1, 512, 16, 16]
        v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)     #[2, 6, 6, 512, 16, 16]

        output = coef_act_exp * v_exp  # (batch,4,channel,size,size) --> [2, 6, 6, 512, 16, 16]

        feat_act = output.sum(1)  # (batch,1,channel,size,size) --> [2, 6, 512, 16, 16]


        # compute connect
        count_coef = coef_act.clone()
        ind = np.diag_indices(self.agent_num)      #(array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 4, 5]))
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0])
        #               non_zero prob(num_connection)           num_agent * batch_size
        # feat_act = [2, 6, 512, 16, 16], coef_act= [2, 6, 6],  num_connect= 0.8333333333333334
        return feat_act, coef_act, num_connect

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat  #torch.Size([12, 512, 16, 16])

    def divide_inputs(self, inputs):
        '''
        Divide the input into a list of several images
        '''
        input_list = []
        for i in range(self.agent_num):
            input_list.append(inputs[:, 3 * i:3 * i + 3, :, :])
        return input_list

    def forward(self, inputs, training=True, MO_flag=False , inference='argmax'):
        batch_size, _, _, _ = inputs.size() #[2, 18, 512, 512]
        input_list = self.divide_inputs(inputs)

        if self.shared_img_encoder == 'unified':
            # vectorize input list
            img_list = []
            for i in range(self.agent_num):
                img_list.append(input_list[i])
            unified_img_list = torch.cat(tuple(img_list), 0) #[12, 3, 512, 512]
            # pass encoder
            feat_maps = self.u_encoder(unified_img_list) #[12, 512, 16, 16]
            # get feat maps for each image
            feat_map = {}
            feat_list = []
            for i in range(self.agent_num):
                feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
                feat_list.append(feat_map[i])
            val_mat = torch.cat(tuple(feat_list), 1)  #[2 - batch size, 6 - agents, 512 - channels, 16, 16 -feat-map size]
        else:
            raise ValueError('Incorrect encoder')

        # pass feature maps through key and query generator
        query_key_maps = self.query_key_net(unified_img_list)  #[12, 256, 4, 4]
        keys = self.key_net(query_key_maps) #[12, 1024]
        if self.has_query:
            querys = self.query_net(query_key_maps) #[12, 32]

        # get key and query
        key = {}
        query = {}
        key_list = []
        query_list = []

        for i in range(self.agent_num):
            key[i] = torch.unsqueeze(keys[batch_size * i:batch_size * (i + 1)], 1)
            key_list.append(key[i])
            if self.has_query:
                query[i] = torch.unsqueeze(querys[batch_size * i:batch_size * (i + 1)], 1)
            else:
                query[i] = torch.ones(batch_size, 1, self.query_size).to('cuda')
            query_list.append(query[i])


        key_mat = torch.cat(tuple(key_list), 1) #[2, 6, 1024]
        query_mat = torch.cat(tuple(query_list), 1) #[2, 6, 32]

        if MO_flag:
            query_mat = query_mat
        else:
            query_mat = torch.unsqueeze(query_mat[:,0,:],1)

        feat_fuse, prob_action = self.attention_net(query_mat, key_mat, val_mat, sparse=self.sparse)
        #feat_fuse --> [2, 6, 512, 16, 16]
        #prob_action --> [2, 6, 6]

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(feat_fuse) #[12, 512, 16, 16]


        pred = self.decoder(feat_fuse_mat) #[12, 11, 512, 512]


        # not related to how we combine the feature (prefer to use the agnets' own frames: to reduce the bandwidth)
        small_bis = torch.eye(prob_action.shape[1])*0.001
        small_bis = small_bis.reshape((1, prob_action.shape[1], prob_action.shape[2]))
        small_bis = small_bis.repeat(prob_action.shape[0], 1, 1).cuda()
        prob_action = prob_action + small_bis


        if training:
            action = torch.argmax(prob_action, dim=1)   #[2, 6]
            num_connect = self.agent_num - 1

            return pred, prob_action, action, num_connect
        else:
            if inference == 'softmax':
                action = torch.argmax(prob_action, dim=1)
                num_connect = self.agent_num - 1

                return pred, prob_action, action, num_connect

            elif inference == 'argmax_test':

                feat_argmax, connect_mat, num_connect = self.argmax_select(val_mat, prob_action)

                feat_argmax_mat = self.agents2batch(feat_argmax)  # (batchsize*agent_num, channel, size, size)
                feat_argmax_mat = feat_argmax_mat.detach()
                pred_argmax = self.decoder(feat_argmax_mat)
                action = torch.argmax(connect_mat, dim=1)
                return pred_argmax, prob_action, action, num_connect

            elif inference == 'activated':
                feat_act, connect_mat, num_connect = self.activated_select(val_mat, prob_action)
                # feat_act = [2, 6, 512, 16, 16], connect_mat = [2, 6, 6], num_connect = 1.416666
                feat_act_mat = self.agents2batch(feat_act)  # (batchsize*agent_num, channel, size, size)
                feat_act_mat = feat_act_mat.detach()

                pred_act = self.decoder(feat_act_mat)

                action = torch.argmax(connect_mat, dim=1)
                return pred_act, prob_action, action, num_connect
            else:
                raise ValueError('Incorrect inference mode')


class MIMO2(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, agent_num=5, shuffle_flag=False, image_size=512,
                 shared_img_encoder=False, key_size=128, query_size=128, enc_backbone='n_segnet_encoder',
                 dec_backbone='n_segnet_decoder'):
        super(MIMO2, self).__init__()

        self.agent_num = agent_num
        self.in_channels = in_channels
        self.shuffle_flag = shuffle_flag
        self.feature_map_channel = 512
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.sparse = sparse

        print('Modified: MIMOcom with learnable token')  # our model: detach the learning of values and keys

        self.token = nn.Parameter(torch.randn(2, in_channels, feat_channel, feat_channel))  # define the learnable token [2, 3, 512, 512]
        print('Token initialized -', self.token.shape)


        self.u_encoder = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                     feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)

        self.key_net = km_generator(out_size=self.key_size, input_feat_sz=image_size / 32)
        self.attention_net = GeneralDotProductAttention(self.query_size, self.key_size)

        # # Message generator
        self.query_key_net = policy_net4(n_classes=n_classes, in_channels=in_channels, enc_backbone=enc_backbone)

        if self.has_query:
            self.query_net = km_generator(out_size=self.query_size, input_feat_sz=image_size / 32)

        # Segmentation decoder
        self.decoder = img_decoder(n_classes=n_classes, in_channels=self.feature_map_channel,
                                   feat_squeezer=feat_squeezer, dec_backbone=dec_backbone)
        self.token_decoder = img_decoder(n_classes=in_channels, in_channels=feat_channel,
                                   feat_squeezer=feat_squeezer, dec_backbone=dec_backbone)
        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        if self.shared_img_encoder == 'unified':
            self.img_net_paras = list(self.u_encoder.parameters()) + list(self.decoder.parameters())

        self.policy_net_paras = list(self.query_key_net.parameters()) + list(
            self.key_net.parameters()) + self.attention_paras
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(self.query_net.parameters())

        self.all_paras = self.img_net_paras + self.policy_net_paras


    def activated_select(self, vals, W_mat, thres=0.155):
        action = torch.mul(W_mat, (W_mat > thres).float())
        attn = action.view(action.shape[0], action.shape[2], 1, 1, 1)  # (batch,5,1,1,1)
        output = attn * vals
        feat_fuse = output.sum(1)  # (batch,1,channel,size,size)

        batch_size = action.shape[0]
        num_connect = torch.nonzero(action[:, :, 1:]).shape[0] / batch_size

        return feat_fuse, action, num_connect

    #modified from srms-Learnwhen2com activated select, aims to eliminate the effect of one noise camera
    def argmin_deselect(self, vals, W_mat): #W_mat: # (batch, 1, size)
        attn_shape = W_mat.shape
        temp = torch.ones(attn_shape).to('cuda')

        min_val, min_ind = torch.min(W_mat, 2)
        for i in range(attn_shape[0]):
            ind = min_ind[i][0].item()
            temp[i][0][ind] = 0

        action = torch.mul(W_mat, (temp > 0).float())

        attn = action.view(action.shape[0], action.shape[2], 1, 1, 1)  # (batch,5,1,1,1)
        output = attn * vals
        feat_fuse = output.sum(1)  # (batch,1,channel,size,size)

        batch_size = action.shape[0]
        num_connect = torch.nonzero(action[:, :, 1:]).shape[0] / batch_size

        return feat_fuse, action, num_connect

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat  # torch.Size([12, 512, 16, 16])

    def divide_inputs(self, inputs):
        '''
        Divide the input into a list of several images
        '''
        input_list = []
        for i in range(self.agent_num):
            input_list.append(inputs[:, 3 * i:3 * i + 3, :, :])
        return input_list

    def forward(self, inputs, training=True, MO_flag = False, inference='argmax'):
        batch_size, _, _, _ = inputs.size()  # [2, 18, 512, 512]
        input_list = self.divide_inputs(inputs)
        num_connect = 0

        if self.shared_img_encoder == 'unified':

            unified_feat_map = torch.cat(
                (input_list[0], input_list[1], input_list[2], input_list[3], input_list[4], input_list[5], self.token),
                0)

            feat_map = self.u_encoder(unified_feat_map)

            feat_map1 = torch.unsqueeze(feat_map[0:batch_size * 1], 1)
            feat_map2 = torch.unsqueeze(feat_map[batch_size * 1:batch_size * 2], 1)
            feat_map3 = torch.unsqueeze(feat_map[batch_size * 2:batch_size * 3], 1)
            feat_map4 = torch.unsqueeze(feat_map[batch_size * 3:batch_size * 4], 1)
            feat_map5 = torch.unsqueeze(feat_map[batch_size * 4:batch_size * 5], 1)
            feat_map6 = torch.unsqueeze(feat_map[batch_size * 5:batch_size * 6], 1)
            token_feat_map = torch.unsqueeze(feat_map[batch_size * 6:batch_size * 7], 1)


        else:
            raise ValueError('Incorrect encoder')

        unified_feat_map = torch.cat(
            (input_list[0], input_list[1], input_list[2], input_list[3], input_list[4], input_list[5], self.token),
            0)
        #[14, 3, 512, 512]
        query_key_map = self.query_key_net(unified_feat_map)


        keys = self.key_net(query_key_map)


        key1 = torch.unsqueeze(keys[0:batch_size * 1], 1)
        key2 = torch.unsqueeze(keys[batch_size * 1:batch_size * 2], 1)
        key3 = torch.unsqueeze(keys[batch_size * 2:batch_size * 3], 1)
        key4 = torch.unsqueeze(keys[batch_size * 3:batch_size * 4], 1)
        key5 = torch.unsqueeze(keys[batch_size * 4:batch_size * 5], 1)
        key6 = torch.unsqueeze(keys[batch_size * 5:batch_size * 6], 1)
        token_key = torch.unsqueeze(keys[batch_size * 6:batch_size * 7], 1)

        if self.has_query:
            querys = self.query_net(query_key_map)

            query1 = torch.unsqueeze(querys[0:batch_size * 1], 1)
            query2 = torch.unsqueeze(querys[batch_size * 1:batch_size * 2], 1)
            query3 = torch.unsqueeze(querys[batch_size * 2:batch_size * 3], 1)
            query4 = torch.unsqueeze(querys[batch_size * 3:batch_size * 4], 1)
            query5 = torch.unsqueeze(querys[batch_size * 4:batch_size * 5], 1)
            query6 = torch.unsqueeze(querys[batch_size * 5:batch_size * 6], 1)
            token_query = torch.unsqueeze(querys[batch_size * 6:batch_size * 7], 1)

            querys = torch.cat((query1, query2, query3, query4, query5, query6, token_query), 1)
        else:
            querys = torch.ones(batch_size, 7, self.query_size).to('cuda')
            token_query = querys[batch_size * 6:batch_size * 7]

        '''
        first self-attention <--- token, agent 1, agent 2, agent 3, agent 4, agent 5, agent 6
        '''
        keys = torch.cat((key1, key2, key3, key4, key5, key6), 1)
        vals = torch.cat((feat_map1, feat_map2, feat_map3, feat_map4, feat_map5, feat_map6), 1)
        querys = torch.cat((query1, query2, query3, query4, query5, query6), 1)

        # aggregate feature to token
        aux_feat, prob_action = self.attention_net(token_query, keys, vals,
                                                   sparse=self.sparse)  # [2, 512, 16, 16], [2, 1, 5]
        #disregard the agent with least action values
        feat_act, _, num_connect = self.argmin_deselect(vals, prob_action)

        #update the token and its key
        updated_token = self.token_decoder(feat_act) #[2, 3, 512, 512]
        token_query_key_map = self.query_key_net(updated_token)
        token_key = self.key_net(token_query_key_map)  #[2, 1024]
        token_key = torch.unsqueeze(token_key, 1)

        featmaps_act = torch.unsqueeze(feat_act, 1)
        #[2, 1, 512, 16, 16]

        #aggregate feature from the learnable token

        aux_feat1, prob_action1 = self.attention_net(query1, torch.cat((key1, token_key), 1),
                                                     torch.cat((feat_map1, featmaps_act), 1),
                                                     sparse=self.sparse)

        aux_feat2, prob_action2 = self.attention_net(query2, torch.cat((key2, token_key), 1),
                                                     torch.cat((feat_map2, featmaps_act), 1),
                                                     sparse=self.sparse)
        aux_feat3, prob_action3 = self.attention_net(query3, torch.cat((key3, token_key), 1),
                                                     torch.cat((feat_map3, featmaps_act), 1),
                                                     sparse=self.sparse)
        aux_feat4, prob_action4 = self.attention_net(query4, torch.cat((key4, token_key), 1),
                                                     torch.cat((feat_map4, featmaps_act), 1),
                                                     sparse=self.sparse)
        aux_feat5, prob_action5 = self.attention_net(query5, torch.cat((key5, token_key), 1),
                                                     torch.cat((feat_map5, featmaps_act), 1),
                                                     sparse=self.sparse)
        aux_feat6, prob_action6 = self.attention_net(query6, torch.cat((key6, token_key), 1),
                                                     torch.cat((feat_map6, featmaps_act), 1),
                                                     sparse=self.sparse)
        num_connect += batch_size * self.agent_num


        feat_fuse = torch.cat((torch.unsqueeze(aux_feat1, 1),
                              torch.unsqueeze(aux_feat2, 1),
                              torch.unsqueeze(aux_feat3, 1),
                              torch.unsqueeze(aux_feat4, 1),
                              torch.unsqueeze(aux_feat5, 1),
                              torch.unsqueeze(aux_feat6, 1)),
                              1)    #[2, 6, 512, 16, 16]


        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(feat_fuse)  # [12, 512, 16, 16]
        pred = self.decoder(feat_fuse_mat)  # [12, 11, 512, 512] confirm if 11 is class-label
        num_connect /= (batch_size * self.agent_num)

        return pred, num_connect



class LearnWhen2Com(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, aux_agent_num=4, shuffle_flag=False, image_size=512,
                 shared_img_encoder=False, key_size=128, query_size=128, enc_backbone='n_segnet_encoder',
                 dec_backbone='n_segnet_decoder'):
        super(LearnWhen2Com, self).__init__()
        # agent_num = 5
        self.aux_agent_num = aux_agent_num
        self.in_channels = in_channels
        self.shuffle_flag = shuffle_flag
        self.feature_map_channel = 512
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.sparse = sparse
        # Encoder
        # Non-shared
        if self.shared_img_encoder == 'unified':
            self.u_encoder = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                         feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
        elif self.shared_img_encoder == 'only_normal_agents':
            self.degarded_encoder = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                                feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
            self.normal_encoder = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                              feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)

        else:
            self.encoder1 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                        feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
            self.encoder2 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                        feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
            self.encoder3 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                        feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
            self.encoder4 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                        feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)
            self.encoder5 = img_encoder(n_classes=n_classes, in_channels=in_channels, feat_channel=feat_channel,
                                        feat_squeezer=feat_squeezer, enc_backbone=enc_backbone)

        # # Message generator
        self.query_key_net = policy_net4(n_classes=n_classes, in_channels=in_channels, enc_backbone=enc_backbone)
        if self.has_query:
            self.query_net = linear(out_size=self.query_size,input_feat_sz=image_size / 32)
        self.key_net = linear(out_size=self.key_size,input_feat_sz=image_size / 32)
        if attention == 'additive':
            self.attention_net = AdditiveAttentin()
        elif attention == 'general':
            self.attention_net = GeneralDotProductAttention(self.query_size, self.key_size)
        else:
            self.attention_net = ScaledDotProductAttention(128 ** 0.5)

        # Segmentation decoder
        self.argmax_decoder = img_decoder(n_classes=n_classes, in_channels=self.feature_map_channel,
                                          agent_num=self.aux_agent_num + 1, dec_backbone=dec_backbone)

        self.decoder = img_decoder(n_classes=n_classes, in_channels=self.feature_map_channel,
                                   feat_squeezer=feat_squeezer, dec_backbone=dec_backbone)
        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        if self.shared_img_encoder == 'unified':
            self.img_net_paras = list(self.u_encoder.parameters()) + list(self.decoder.parameters())
        elif self.shared_img_encoder == 'only_normal_agents':
            self.img_net_paras = list(self.degarded_encoder.parameters()) + list(
                self.normal_encoder.parameters()) + list(self.decoder.parameters())
        else:
            self.img_net_paras = list(self.encoder1.parameters()) + \
                                 list(self.encoder2.parameters()) + \
                                 list(self.encoder3.parameters()) + \
                                 list(self.encoder4.parameters()) + \
                                 list(self.encoder5.parameters()) + \
                                 list(self.decoder.parameters())

        self.img_net_paras = self.img_net_paras + list(self.argmax_decoder.parameters())

        self.policy_net_paras = list(self.query_key_net.parameters()) + list(
            self.key_net.parameters()) + self.attention_paras
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(self.query_net.parameters())


        self.all_paras = self.img_net_paras + self.policy_net_paras

    def divide_inputs(self, inputs):
        '''
        Divide the input into a list of several images
        '''
        input_list = []
        divide_num = 5
        for i in range(divide_num):
            input_list.append(inputs[:, 3 * i:3 * i + 3, :, :])

        return input_list

    def shape_feat_map(self, feat, size):
        return torch.unsqueeze(feat.view(-1, size[1] * size[2] * size[3]), 1)

    def argmax_select(self, feat_map1, feat_map2, feat_map3, feat_map4, feat_map5, action, batch_size):

        num_connect = 0
        feat_list = []
        for k in range(batch_size):
            if action[k] == 0:
                feat_list.append(torch.unsqueeze(feat_map1[k], 0))
            elif action[k] == 1:
                feat_list.append(torch.unsqueeze(feat_map2[k], 0))
                num_connect = num_connect + 1
            elif action[k] == 2:
                feat_list.append(torch.unsqueeze(feat_map3[k], 0))
                num_connect = num_connect + 1
            elif action[k] == 3:
                feat_list.append(torch.unsqueeze(feat_map4[k], 0))
                num_connect = num_connect + 1
            elif action[k] == 4:
                feat_list.append(torch.unsqueeze(feat_map5[k], 0))
                num_connect = num_connect + 1
            else:
                raise ValueError('Incorrect action')
        num_connect = num_connect / batch_size

        feat_argmax = tuple(feat_list)
        feat_argmax = torch.cat(feat_argmax, 0)
        feat_argmax = torch.squeeze(feat_argmax, 1)
        return feat_argmax, num_connect

    def activated_select(self, vals, W_mat, thres=0.2):
        action = torch.mul(W_mat, (W_mat > thres).float())
        attn = action.view(action.shape[0], action.shape[2], 1, 1, 1)  # (batch,5,1,1,1)
        output = attn * vals
        feat_fuse = output.sum(1)  # (batch,1,channel,size,size)

        batch_size = action.shape[0]
        num_connect = torch.nonzero(action[:, :, 1:]).shape[0] / batch_size

        return feat_fuse, action, num_connect

    def forward(self, inputs, training=True, inference='argmax'):
        batch_size, _, _, _ = inputs.size()
        input_list = self.divide_inputs(inputs)
        if self.shared_img_encoder == 'unified':
            unified_feat_map = torch.cat((input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]), 0)  #[10, 3, 512, 512]\

            feat_map = self.u_encoder(unified_feat_map)  #[10, 512, 16, 16]

            feat_map1 = feat_map[0:batch_size * 1]
            feat_map2 = feat_map[batch_size * 1:batch_size * 2]
            feat_map3 = feat_map[batch_size * 2:batch_size * 3]
            feat_map4 = feat_map[batch_size * 3:batch_size * 4]
            feat_map5 = feat_map[batch_size * 4:batch_size * 5]

        elif self.shared_img_encoder == 'only_normal_agents':
            feat_map1 = self.degarded_encoder(input_list[0])
            unified_normal_feat_map = torch.cat((input_list[1], input_list[2], input_list[3], input_list[4]), 0)
            feat_map = self.normal_encoder(unified_normal_feat_map)
            feat_map2 = feat_map[0:batch_size * 1]
            feat_map3 = feat_map[batch_size * 1:batch_size * 2]
            feat_map4 = feat_map[batch_size * 2:batch_size * 3]
            feat_map5 = feat_map[batch_size * 3:batch_size * 4]
        else:
            feat_map1 = self.encoder1(input_list[0])
            feat_map2 = self.encoder2(input_list[1])
            feat_map3 = self.encoder3(input_list[2])
            feat_map4 = self.encoder4(input_list[3])
            feat_map5 = self.encoder5(input_list[4])

        unified_feat_map = torch.cat((input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]), 0)    #[10, 3, 512, 512]

        query_key_map = self.query_key_net(unified_feat_map)    #[10, 256, 4, 4]

        query_key_map1 = query_key_map[0:batch_size * 1]

        keys = self.key_net(query_key_map) #[10, 1024]

        key1 = torch.unsqueeze(keys[0:batch_size * 1], 1)
        key2 = torch.unsqueeze(keys[batch_size * 1:batch_size * 2], 1)
        key3 = torch.unsqueeze(keys[batch_size * 2:batch_size * 3], 1)
        key4 = torch.unsqueeze(keys[batch_size * 3:batch_size * 4], 1)
        key5 = torch.unsqueeze(keys[batch_size * 4:batch_size * 5], 1)


        if self.has_query:
            querys = self.query_net(query_key_map) #[10, 8]
            query = torch.unsqueeze(querys[0:batch_size * 1], 1) #[2, 1, 8]

        else:
            query = torch.ones(batch_size, 1, self.query_size).to('cuda')

        feat_map1 = torch.unsqueeze(feat_map1, 1)  # (batch,1,channel,size,size)
        feat_map2 = torch.unsqueeze(feat_map2, 1)  # (batch,1,channel,size,size)
        feat_map3 = torch.unsqueeze(feat_map3, 1)  # (batch,1,channel,size,size)
        feat_map4 = torch.unsqueeze(feat_map4, 1)  # (batch,1,channel,size,size)
        feat_map5 = torch.unsqueeze(feat_map5, 1)  # (batch,1,channel,size,size)

        keys = torch.cat((key1, key2, key3, key4, key5), 1)  # [2, 5, 1024]

        vals = torch.cat((feat_map1, feat_map2, feat_map3, feat_map4, feat_map5), 1)  # [2, 5, 512, 16, 16])

        #query [2, 1, 8] keys [2, 6, 1024] val [2, 6, 512, 16, 16]
        aux_feat, prob_action = self.attention_net(query, keys, vals,
                                                   sparse=self.sparse)  # [2, 512, 16, 16], [2, 1, 5]

        pred = self.decoder(aux_feat) #[2, 11, 512, 512]

        if training:
            action = torch.argmax(prob_action, dim=2)
            return pred, prob_action, action
        else:
            if inference == 'softmax':
                action = torch.argmax(prob_action, dim=2)
                num_connect = 4
                return pred, prob_action, action, num_connect
            elif inference == 'argmax_test':
                action = torch.argmax(prob_action, dim=2)
                feat_argmax, num_connect = self.argmax_select(feat_map1, feat_map2, feat_map3, feat_map4, feat_map5,
                                                              action, batch_size)
                featmaps_argmax = feat_argmax.detach()
                pred_argmax = self.decoder(featmaps_argmax)
                return pred_argmax, prob_action, action, num_connect
            elif inference == 'activated':
                feat_act, action, num_connect = self.activated_select(vals, prob_action)
                featmaps_act = feat_act.detach()
                pred_act = self.decoder(featmaps_act)
                return pred_act, prob_action, action, num_connect
            else:
                raise ValueError('Incorrect inference mode')


