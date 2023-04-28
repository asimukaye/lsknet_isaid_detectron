import torch
from collections import OrderedDict
import numpy as np


main_dict = torch.load('/home/asim.ukaye/cv_proj/lsknet_isaid_custom/pretrained_weights/lsk_t_fpn_1x_dota_le90_20230206-3ccee254.pth')

backbone = torch.load('/home/asim.ukaye/cv_proj/lsknet_isaid_custom/pretrained_weights/lsk_t_backbone-2ef8a593.pth')

imagenet_out_dict = OrderedDict({})
dota_out_dict = OrderedDict({})

for k, v in backbone['state_dict'].items():
    if 'head' in k:
        continue
    else:
        imagenet_out_dict[k] = v

        dota_out_dict[k] = main_dict['state_dict']['backbone.'+k]

dota_pretrained_path = '/home/asim.ukaye/cv_proj/lsknet_isaid_custom/pretrained_weights/lsknet_t_dota_pretrained.pth'
imagenet_pretrained_path = '/home/asim.ukaye/cv_proj/lsknet_isaid_custom/pretrained_weights/imagenet_t_pretrained.pth'

# print('dota_out: ', dota_out_dict.keys())

torch.save({'state_dict':dota_out_dict},dota_pretrained_path)
torch.save({'state_dict':imagenet_out_dict}, imagenet_pretrained_path)


dota_dict = torch.load(dota_pretrained_path)

imgnet_dict = torch.load(imagenet_pretrained_path)

model_load = '/home/asim.ukaye/cv_proj/output/ResNet50FPN_iSAID/model_0004999.pth'

# print(imgnet_dict.keys())
sample = torch.load(model_load)

out_sample_tail = OrderedDict({})


for k, v in sample['model'].items():
    if 'backbone.bottom_up' in k:
        continue
    else:
        out_sample_tail[k] = v

out_prepend = OrderedDict({})
# for k, v in dota_dict.items():
for k, v in imgnet_dict['state_dict'].items():
    out_prepend['backbone.bottom_up.'+ k] = v

# print('out_prepend: ', out_prepend.keys())

out_prepend.update(out_sample_tail)
# print('out_prepend_ipdat: ', out_prepend.keys())

sample_clone = sample.copy()
sample_clone['model'] = out_prepend

torch.save(out_prepend, 'imagenet_lsknet_t_fused.pth')

out_prepend_dota = OrderedDict({})
for k, v in dota_dict['state_dict'].items():
    out_prepend_dota['backbone.bottom_up.'+ k] = v
out_prepend_dota.update(out_sample_tail)

sample_clone['model'] = out_prepend_dota

torch.save(out_prepend_dota, 'dota_lsknet_t_fused.pth')

