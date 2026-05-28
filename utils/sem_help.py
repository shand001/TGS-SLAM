
import torch
def label2map(label_gt, num_semantic=-1,device="cuda"):
    # label_gt [1,h,w]
    # label_map [num_classes,h,w]
    
    height, width = label_gt.shape[1], label_gt.shape[2]
    if num_semantic<0:
        max_class = torch.max(label_gt).item()
        num_classes = max_class+1
        num_classes = int(num_classes)
    else:
        num_classes = num_semantic

    label_map_list = []
    for i_sem in range(0,num_classes):
        label_map_single = torch.zeros((1, height, width), dtype=torch.float, device=device)
        label_map_single[label_gt==i_sem] = 1.0
        label_map_single = label_map_single.view(1,height, width)
        label_map_list.append(label_map_single)

    label_map = torch.cat(label_map_list,dim=0)
    # print("label_map shape:",label_map.shape)  # [num_classes,h,w]
    # print("num_classes:",num_classes)
    return label_map, num_classes  #(52, 680, 1200)