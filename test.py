import torch
import torchvision




def test_drwkv_model():
    from models_drwkv import DRWKV_models
    from thop import profile

    for k, v in DRWKV_models.items(): 
        print(k)
        model = v(img_size=32).cuda()
        input_image = torch.randn(1, 3, 32, 32).cuda()
        times_steps = torch.randint(1, 100, (1,)).cuda()
        flops, _ = profile(model, inputs=(input_image, times_steps ))
        # out = model(x=input_image, timesteps=times_steps)
        #print(out.size())
        print('FLOPs = ' + str(flops * 2/1000**3) + 'G')
        
        parameters_sum = sum(x.numel() for x in model.parameters())
        print(parameters_sum / 1000000.0, "M")


def test_cifar10(): 
    data_path = "/TrainData/Multimodal/zhengcong.fei/dis/data"
    cifar10 = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False
    )
    print(cifar10)
    print(cifar10_test[0])



def test_imagenet1k(): 
    data_path = '/maindata/data/shared/multimodal/public/dataset_img_only/imagenet/data/train' 
    import torchvision.datasets as datasets
    dataset_train = datasets.ImageFolder(data_path) 
    print(dataset_train[0])


def imagenet_formation():
    data_path = '/maindata/data/shared/multimodal/public/dataset_img_only/imagenet/data/train_org'
    from tqdm import tqdm 
    import os 
    import shutil 
    img_list = os.listdir(data_path) 
    
    print(len(img_list)) 
    class_list = []
    for img in img_list: 
        class_name = img.split('_')[0]
        class_list.append(class_name)
    class_list = set(class_list)
    print(len(class_list))

    target_path = '/maindata/data/shared/multimodal/public/dataset_img_only/imagenet/data/train'
    for class_name in class_list: 
        directory = os.path.join(target_path, class_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    for img in tqdm(img_list): 
        class_name = img.split('_')[0]
        src_path = os.path.join(data_path, img)
        tgt_path = os.path.join(target_path, class_name, img)
        shutil.move(src_path, tgt_path)



def test_celeba(): 
    from datasets import load_dataset
    data_path = "/TrainData/Multimodal/zhengcong.fei/dis/data/CelebA"
    dataset = load_dataset(data_path) 
    # dataset = dataset['train']
    # dataset = dataset.map(lambda e: e['image'].convert('RGB'), batched=True)
    #print(dataset[0])
    print(dataset['train'][0].keys())
    #print(dataset['train'][0]['image'].convert("RGB"))
    # print(len(dataset['train']))


def test_fid_score(): 
    from tools.fid_score import calculate_fid_given_paths 
    path1 = '/TrainData/Multimodal/zhengcong.fei/dis/results/cond_cifar10_small/his'
    path2 = '/TrainData/Multimodal/zhengcong.fei/dis/results/uncond_cifar10_small/his'
    fid = calculate_fid_given_paths((path1, path2))



def test_vae(): 
    from diffusers.models import AutoencoderKL 
    vae_path = '/TrainData/Multimodal/zhengcong.fei/dis/vae'
    vae = AutoencoderKL.from_pretrained(vae_path)



def test_rwkv(): 
    from models_drwkv import DiffRWKVModel 
    model = DiffRWKVModel().cuda()
    input_image = torch.randn(1, 3, 64, 64).cuda()
    times_steps = torch.randint(1, 100, (1,)).cuda()

    output = model(input_image, timesteps=times_steps)


test_drwkv_model()
# test_cifar10()
# test_imagenet1k()
# test_celeba()
# test_fid_score()
# test_vae()