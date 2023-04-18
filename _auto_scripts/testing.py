import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
import utils
import torch
import mymodels 
import mydataset 
import numpy as np
import pandas as pd
import pathlib
import yaml
import argparse
import tqdm

FOLDER_PATH = '/home/suncheol/code/FedTest/0_fedmhad/checkpoints'
EXCEL_PATH = 'result.xlsx'
if globals().get('__file__') is None:
    __file__ = 'testing.py'
    
parent_path = pathlib.Path(__file__).parent.parent
yamlfilepath = parent_path.joinpath('config.yaml')
args = yaml.load(yamlfilepath.open('r'), Loader=yaml.FullLoader)
args = argparse.Namespace(**args)
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

if args.dataset == 'cifar10':
    publicdata = 'cifar100'
    args.N_class = 10
elif args.dataset == 'cifar100':
    publicdata = 'imagenet'
    args.N_class = 100
elif args.dataset == 'pascal_voc2012':
    publicdata = 'mscoco'
    args.N_class = 20
    
path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012')
test_imgs = np.load(path.joinpath('PASCAL_VOC_val_224_Img.npy'))
test_labels = np.load(path.joinpath('PASCAL_VOC_val_224_Label.npy'))
test_imgs, test_labels = mydataset.data_cifar.filter_images_by_label_type(args.task, test_imgs, test_labels)
test_dataset = mydataset.data_cifar.mydataset(test_imgs, test_labels)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, sampler=None)

def find_model_files(folder_path):
    """
    주어진 폴더 경로 내에 있는 .pt 또는 .pth 파일들을 반환합니다.
    """
    model_files = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.pt') or filename.endswith('.pth'):
                model_files.append(os.path.join(dirpath, filename))
    return model_files

def test_model(model_path, test_loader):
    """
    주어진 모델 파일을 테스트하고, 결과를 반환합니다.
    """
    try:
        model = mymodels.define_model(modelname=args.model_name, num_classes=args.N_class, pretrained=args.pretrained)
        utils.load_dict(model_path, model)
        model.eval()
        # testacc = utils.AverageMeter()
        m = torch.nn.Sigmoid()
        output_list = []
        target_list = []

        with torch.no_grad():
            for i, (images, target, _) in enumerate(test_loader):
                images = images.cuda()
                target = target.cuda()
                output = model(images)
                
                output_list.append(m(output).detach().cpu().numpy())
                target_list.append(target.detach().cpu().numpy())
                
                # testacc.update(acc)
        output = np.concatenate(output_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        acc, = utils.accuracy(output, target)
        top_k = utils.multi_label_top_margin_k_accuracy(target, output, margin=0)
        mAP, _ = utils.compute_mean_average_precision(target, output)
        acc, top_k, mAP = round(acc, 4), round(top_k, 4), round(mAP, 4)
        print(f'path: {model_path}, acc: {acc}, top_k: {top_k}, mAP: {mAP}')
    except:
        print(f'Error occured while testing {model_path}. Skipping...')
        return {'model_path': model_path, 'acc': None, 'top_k': None, 'mAP': None}
    return {'model_path': model_path, 'acc': acc, 'top_k': top_k, 'mAP': mAP}

def update_excel_file(excel_path, results):
    """
    주어진 결과를 엑셀 파일에 추가합니다.
    """
    # 엑셀 파일 로드
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=['model_path', 'acc', 'top_k', 'mAP'])

    # 새로운 결과 추가
    for result in results:
        # 이미 기록된 파일은 건너뜁니다.
        if result['model_path'] in df['model_path'].tolist():
            continue
        df = df.append(result, ignore_index=True)

    # 결과를 엑셀 파일로 저장합니다.
    df.to_excel(excel_path, index=False)

def main(folder_path, excel_path):
    """
    주어진 폴더 경로 내의 모델들을 테스트하고, 결과를 엑셀 파일에 추가합니다.
    """
    while True:
        # 새로운 모델 파일 찾기
        model_files = find_model_files(folder_path)

        # 모델 테스트 실행
        results = []
        for model_file in model_files:
            result = test_model(model_file, test_loader)
            results.append(result)

        # 결과를 엑셀 파일에 추가
        if results:
            update_excel_file(excel_path, results)

        # 10초마다 반복 실행
        time.sleep(10)

if __name__ == '__main__':
    
    main(FOLDER_PATH, EXCEL_PATH)