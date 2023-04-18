import os
import numpy as np
import logging
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import utils
import copy
import random
import loss as Loss
import mydataset
from tqdm import tqdm

class FedMAD:
    def __init__(self, central, distil_loader, private_data, val_loader, 
                   writer, experiment, args, initpth=True, localmodel=None):
        # import ipdb; ipdb.set_trace()
        self.localmodel = localmodel
        self.central = central
        self.N_parties = args.N_parties
        self.distil_loader = distil_loader
        self.private_data = private_data
        self.val_loader = val_loader
        self.writer = writer
        self.args = args
        self.experiment = experiment
        self.prevacc = 0
        self.prev_model_state_dict = {}
        
        # ensemble and loss
        self.totalN = self.get_totalN(self.N_parties, self.private_data)
        self.countN = self.get_class_counts(self.N_parties, self.args.N_class, self.private_data)
        self.locallist= list(range(0, self.N_parties))# local number list
        self.clscnt = args.clscnt# if ensemble is weighted by number of local sample
        self.voteout = args.voteout
        if args.lossmode == 'l1':
            self.criterion = torch.nn.L1Loss(reduce=True)
        elif args.lossmode=='kl': 
            # self.criterion = Loss.kl_loss(T=3, singlelabel=args.singlelabel)
            self.criterion = Loss.kl_loss(T=3, singlelabel=True)
        else:
            raise ValueError
            
        if args.sublossmode=='at':
            self.sub_criterion = Loss.at_loss()
        elif args.sublossmode=='mha':
            self.sub_criterion = Loss.MHALoss(distill_heads=args.distill_heads)
        else:
            self.sub_criterion = None
        
        # distill optimizer
        self.bestacc = 0
        self.best_statdict = self.central.state_dict()
        
        # import ipdb; ipdb.set_trace()
        #path to save ckpt
        noise = 'noisy' if args.noise else 'clean'
        exclude = f'excluded_{args.exclude_heads}' if args.exclude_heads > 0 else ''
        self.rootdir = f'./checkpoints/{args.dataset}/{args.model_name}_{args.task}_{noise}_{args.dirichlet:.1f}_{args.dis_lr}/{exclude}/a{args.alpha}+sd{args.seed}+e{args.initepochs}+b{args.batchsize}+l{args.lossmode}+sl{args.sublossmode}'
        if not os.path.isdir(self.rootdir):
            os.makedirs(self.rootdir)
        if initpth:
            if not args.subpath:
                if args.oneshot:
                    args.subpath = f'oneshot_c{args.C}_q{args.quantify}_n{args.noisescale}_h{args.distill_heads}'
                else:
                    args.subpath = f'oneshot_c{args.C}_q{args.quantify}_n{args.noisescale}_h{args.distill_heads}'
            self.savedir = os.path.join(self.rootdir, args.subpath)
            if not os.path.isdir(self.savedir):
                os.makedirs(self.savedir)
        self.init_locals(initpth, init_dir='')
    
    def get_totalN(self, N_parties, private_data):
        '''
        args:
            N_parties: number of parties
            private_data: list of private data
        return:
            totalN: (N_parties)
        '''        
        totalN = []
        for n in range(N_parties):
            totalN.append(private_data[n]['x'].shape[0])
        totalN = torch.tensor(totalN).cuda() #nlocal
        return totalN
        
    def get_class_counts(self, N_parties, N_class, private_data):
        '''
        args:
            N_parties: number of parties
            N_class: number of classes
            private_data: list of private data
        return:
            countN: (N_parties, N_class)
        '''
        countN = np.zeros((N_parties, N_class))
        for n in range(N_parties):
            counterclass = np.sum(private_data[n]['y'], axis=0)
            for m in range(N_class):
                countN[n][m] = counterclass[m]
        return countN
         
    def init_locals(self, initpth=True, init_dir=''):
        epochs = self.args.initepochs
        if self.localmodel is None:
            self.localmodels = utils.copy_parties(self.N_parties, self.central)
        else:
            self.localmodels = utils.copy_parties(self.N_parties, self.localmodel)
        if not init_dir:
            init_dir = self.rootdir
            print('init_dir : ', init_dir)
        if initpth:
            for n in range(self.N_parties):
                # savename = os.path.join(init_dir, str(n)+'.pt')
                savename = os.path.join(init_dir, f'model-{n}.pth')
                if os.path.exists(savename):
                    #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
                    logging.info(f'Loading Local{n}......')
                    print('filepath : ', savename)
                    self.localmodels[n].module.setExcludedHead([i for i in range(self.args.exclude_heads)])
                    utils.load_dict(savename, self.localmodels[n])
                    # result = self.validate_model(self.localmodels[n])
                    acc = 0.0
                else:
                    logging.info(f'Init Local{n}, Epoch={epochs}......')
                    acc = self.trainLocal(savename, modelid=n)
                logging.info(f'Init Local{n}--Epoch={epochs}, Acc:{(acc):.2f}')
     
    def init_central(self):
        initname = os.path.join(self.rootdir, self.args.initcentral)
        if os.path.exists(initname):
            utils.load_dict(initname, self.central)
            acc = self.validate_model(self.central)['acc']
            logging.info(f'Init Central--Acc:{(acc):.2f}')
            self.bestacc = acc
            self.best_statdict = copy.deepcopy(self.central.state_dict())
        else:
            raise ValueError

    def getTotalGradcam(self, model, images, selectN, usecentral=True):
        total_gradcam = [] 
        for n in selectN:
            tmodel = copy.deepcopy(self.localmodels[n])
            gradcam = tmodel.module.get_class_activation_map(images, y=None)
            total_gradcam.append(gradcam)
            del tmodel
        return total_gradcam

    def get_multi_head_attention_map(self, model, images):
        '''
        args:
            model: local model
            images: batch * 3 * 224 * 224
        return:
            mha: batch * nhead * 224 * 224
        '''
        tmodel = copy.deepcopy(model)
        mha = tmodel.module.get_attention_maps_postprocessing(images)
        del tmodel
        return mha
    
    def get_total_multi_head_attention_map(self, model, images, selectN):
        '''
        args:
            model: central model
            images: batch * 3 * 224 * 224
            selectN: list of client number
        return:
            total_mha: n_client * batch * nhead * 224 * 224
        '''
        return torch.stack([self.get_multi_head_attention_map(self.localmodels[n], images) for n in selectN]) # n_client * batch * nhead * 224 * 224

    def get_total_logits(self, model, images, selectN, usecentral=True):
        if usecentral:
            total_logits, total_attns = self.central(images).detach()
        else:
            total_logits = []
            total_attns = []
            for n in selectN:
                tmodel = copy.deepcopy(self.localmodels[n])
                # logits = torch.sigmoid(tmodel(images).detach())
                logits, attn = tmodel(images, return_attn=True)
                logits = torch.sigmoid(logits).detach()
                total_logits.append(logits)
                total_attns.append(attn.detach())
                del tmodel
            total_logits = torch.stack(total_logits)
            total_attns = torch.stack(total_attns)
        return total_logits, total_attns
        
    def compute_class_weights(self, class_counts):
        """
        Args:
            class_counts (torch.Tensor): (num_samples, num_classes)
        Returns:
            class_weights (torch.Tensor): (num_samples, num_classes)
        """
        # Normalize the class counts per sample
        class_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
        return class_weights
        
    def compute_ensemble_logits(self, client_logits, class_weights):
        """
        Args:
            client_logits (torch.Tensor): (num_samples, batch_size, num_classes)
            class_weights (torch.Tensor): (num_samples, num_classes)
        Returns:
            ensemble_logits (torch.Tensor): (batch_size, num_classes)
        """
        weighted_logits = client_logits * class_weights.unsqueeze(1)  # (num_samples, batch_size, num_classes)
        sum_weighted_logits = torch.sum(weighted_logits, dim=0)  # (batch_size, num_classes)
        sum_weights = torch.sum(class_weights, dim=0)  # (num_classes)
        ensemble_logits = sum_weighted_logits / sum_weights
        return ensemble_logits

    def get_ensemble_logits(self, total_logits, selectN, logits_weights):
        # class_counts = self.countN[selectN]
        # class_weights = self.compute_class_weights(torch.from_numpy(class_counts).float().cuda())
        if self.voteout:
            ensemble_logits = Loss.weight_psedolabel(total_logits, self.countN[selectN], noweight=True).detach()
        else:
            ensemble_logits = self.compute_ensemble_logits(total_logits, logits_weights)
        return ensemble_logits

    def compute_euclidean_norm(self, vector_a, vector_b):
        return torch.tensor(1) - torch.sqrt(torch.sum((vector_a - vector_b) ** 2, dim=-1))

    def compute_cosine_similarity(self, vector_a, vector_b):
        # print(vector_a.shape, vector_b.shape)
        cs = torch.sum(vector_a * vector_b, dim=-1) / (torch.norm(vector_a, dim=-1) * torch.norm(vector_b, dim=-1))
        return cs

    def calculate_normalized_similarity_weights(self, target_vectors, client_vectors, similarity_method='euclidean'):
        if similarity_method == 'euclidean':
            similarity_function = self.compute_euclidean_distance
        elif similarity_method == 'cosine':
            similarity_function = self.compute_cosine_similarity
        else:
            raise ValueError("Invalid similarity method. Choose 'euclidean' or 'cosine'.")

        target_vectors_expanded = target_vectors.unsqueeze(0)  # Shape: (1, batch_size, n_class)
        
        similarities = similarity_function(target_vectors_expanded, client_vectors)  # Shape: (n_client, batch_size)
        mean_similarities = torch.mean(similarities, dim=1)  # Shape: (n_client)
        normalized_similarity_weights = mean_similarities / torch.sum(mean_similarities)  # Shape: (n_client)
        # print("normalized_similarity_weights", normalized_similarity_weights)
        # print(normalized_similarity_weights)
        return normalized_similarity_weights

    def get_logit_weights(self, total_logits, labels, selectN, method='ap'):
        if method == 'ap':
            import metrics
            ap_list = []
            for i in range(total_logits.shape[0]):
                client_logits = total_logits[i].detach().cpu().numpy()
                map, aps = metrics.compute_mean_average_precision(labels, client_logits)
                ap_list.append(aps)
            ap_list = np.array(ap_list)
            ap_list = torch.from_numpy(ap_list).float().cuda()
            ap_weights = ap_list / ap_list.sum(dim=0, keepdim=True)
            return ap_weights
        elif method == 'count':
            class_counts = self.countN[selectN]
            class_counts = torch.from_numpy(class_counts).float().cuda()
            class_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
            # class_weights = self.compute_class_weights(torch.from_numpy(class_counts).float().cuda())
            return class_weights
        else :
            raise ValueError("Invalid weight method. Choose 'ap' or 'count'.")
    
    def distill_onemodel_batch(self, model, images, labels, selectN, optimizer, usecentral=True):
        ''' for ensemble logits '''
        with torch.autograd.set_detect_anomaly(True):
            total_logits, total_attns = self.get_total_logits(model, images, selectN, usecentral=usecentral)
            logits_weights = self.get_logit_weights(total_logits, labels, selectN)
            ensemble_logits = self.get_ensemble_logits(total_logits, selectN, logits_weights) * (1 - self.args.lambda_kd)
            sim_weights = self.calculate_normalized_similarity_weights(ensemble_logits, total_logits, "cosine")
            
            model.train()
            central_logits, central_attn = model(images, return_attn=True)
            central_logits = torch.sigmoid(central_logits)
            logit_loss = self.criterion(central_logits, ensemble_logits)
            
            if self.args.sublossmode=='at':
                total_gradcam = self.getTotalGradcam(model, images, selectN, usecentral=usecentral)
                union, intersection = Loss.weight_gradcam(total_gradcam, self.countN)
                gradcam = model.module.get_class_activation_map(images, y=None)
                sub_loss = self.sub_criterion(union, intersection, gradcam) * self.args.lambda_kd
            elif self.args.sublossmode=='mha':
                # union, intersection = self.sub_criterion.get_union_intersection_image(total_attns)
                # # total_mha = self.get_total_multi_head_attention_map(model, images, selectN)
                # # mha = model.module.get_attention_maps_postprocessing(images)
                # sub_loss = self.sub_criterion(union, intersection, central_attn)
                # print('sim_weights : ', sim_weights)
                sub_loss = self.sub_criterion(total_attns, central_attn, sim_weights) * self.args.lambda_kd
            else : 
                sub_loss = torch.tensor(0.0).cuda()
            
            total_loss = logit_loss + sub_loss
            # print('logit_loss : ', logit_loss, 'sub_loss : ', sub_loss, 'total_loss : ', total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return logit_loss, sub_loss, ensemble_logits.detach(), central_logits.detach()
    
    def get_optimizer(self, model):
        args = self.args
        last_layer_name = list(model.module.named_children())[-1][0]
        parameters = [
            {'params': [p for n, p in model.module.named_parameters() if last_layer_name not in n], 'lr': args.lr},
            {'params': [p for n, p in model.module.named_parameters() if last_layer_name in n], 'lr': args.lr*100},
        ]
        if self.args.optim == 'SGD':
            optimizer = optim.SGD( params= parameters, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wdecay)
        else:
            optimizer = optim.Adam( params= parameters, lr=self.args.lr, betas=(self.args.momentum, 0.999), weight_decay=self.args.wdecay)
        return optimizer
    
    def train_all_local_clients_with_distill(self, nRound, distill_loader, ensemble_logits, total_attn_list):
        for i in range(self.args.N_parties):
            self.train_local_clients_with_distill(i, nRound, distill_loader, ensemble_logits, total_attn_list)
    
    def train_local_clients_with_distill(self, nModel, nRound, distill_loader, ensemble_logits, total_attn_list):
        model = self.localmodels[nModel]
        model.train()
        optimizer = self.get_optimizer(model)
        m = torch.nn.Sigmoid()
        for batch_idx, (images, _, _) in tqdm(enumerate(distill_loader)):
            images = images.cuda()
            logits, attn = model(images, return_attn=True)
            logits = m(logits)
            logit_loss = self.criterion(logits, ensemble_logits[batch_idx].detach().cuda())
            # print('logit_loss : ', logit_loss)
            optimizer.zero_grad()
            logit_loss.backward()
            optimizer.step()
        
        train_dataset = mydataset.data_cifar.mydataset(self.private_data[nModel]['x'], self.private_data[nModel]['y'], train=True, verbose=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batchsize, shuffle=True) 
        for batch_idx, (images, target, _) in tqdm(enumerate(train_loader)):
            images = images.cuda()
            target = target.cuda()
            logits, attn = model(images, return_attn=True)
            logits = m(logits)
            logit_loss = self.criterion(logits, target)
            # print('logit_loss : ', logit_loss)
            optimizer.zero_grad()
            logit_loss.backward()
            optimizer.step()
        
        result = self.evaluate_validation_metrics(model)
        result = {f'client{nModel}_' + k: v for k, v in result.items()}
        # self.experiment.log_metrics(result, step=nRound)
        return result
    
    def distill_local_central(self):
        args = self.args
        net = self.central
        optimizer = self.get_optimizer(net)    
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.fedrounds), eta_min=self.args.dis_lrmin,)
        
        savename = os.path.join(self.savedir, f'q{args.quantify}_n{args.noisescale}_{args.optim}_b{args.disbatchsize}_{args.dis_lr}_{args.fedrounds}_{args.dis_lrmin}_m{args.momentum}')
        bestacc = self.bestacc
        bestname = ''
        selectN = self.locallist        
        step = 0
        acc = 0.0
        
        self.totalSteps = int(args.fedrounds)
        self.earlystopSteps = int(10)
        earlyStop = utils.EarlyStop(self.earlystopSteps, self.totalSteps, bestacc=self.bestacc)
        
        for epoch in range(0, args.fedrounds):
            loss_avg = 0.0
            sub_loss_avg = 0.0
            ensemble_logit_list = []
            total_attn_list = []
            for i, (images, labels, _) in tqdm(enumerate(self.distil_loader)):
                images = images.cuda()
                if self.args.C<1:
                    selectN = random.sample(self.locallist, int(args.C*self.N_parties))
                
                loss, subloss, ensemble_logits, total_attns = self.distill_onemodel_batch(self.central, images, labels, selectN, optimizer, usecentral=False) 
                ensemble_logit_list.append(ensemble_logits)
                # total_attn_list.append(total_attns)
                loss_avg += loss.item()
                sub_loss_avg += subloss.item()
                step += 1
            
            loss_avg /= len(self.distil_loader)
            sub_loss_avg /= len(self.distil_loader)
            
            if epoch%1==0:
                result = self.validate_model(self.central)
                acc = result['acc']
                 
            logging.info(f'Epoch{epoch}, Acc{(acc):.4f}')
            stop, best = earlyStop.update(epoch, acc)
            if best:
                if bestname:
                    os.system(f'rm {bestname}')
                bestacc = acc
                bestname = f'{savename}_e{epoch}_{(bestacc):.4f}.pt'
                torch.save(self.central.state_dict(), bestname)
                logging.info(f'========Best...Epoch{epoch}, Acc{(acc):.4f}')
            else:
                logging.info(f'Epoch{epoch}, Acc{(acc):.4f}')
            
            if self.writer is not None:
                self.writer.add_scalar('loss', loss, epoch, display_name='loss')
                self.writer.add_scalar('at_loss', subloss, epoch, display_name='at_loss')
                self.writer.add_scalar('DisACC', acc, epoch, display_name='DisACC')
                self.writer.add_scalar('BestACC', bestacc, epoch, display_name='BestACC')
            if self.experiment is not None:
                self.experiment.log_metric('lg_loss', loss_avg, step=epoch)
                self.experiment.log_metric('at_loss', sub_loss_avg, step=epoch)
                self.experiment.log_metric('DisACC', acc, step=epoch)
                self.experiment.log_metric('BestACC', bestacc, step=epoch)
                self.experiment.log_metric('top_k', result['top_k'], step=epoch)
                self.experiment.log_metric('mAP', result['mAP'], step=epoch)
                self.experiment.log_metric('mROC', result['mROC'], step=epoch)
            
            scheduler.step()
                         
            if stop:
                logging.info(f'Early Stop at Epoch{epoch}, Acc{(acc):.4f}')
                break
            
            if epoch%2==0:
                self.train_all_local_clients_with_distill(epoch, self.distil_loader, ensemble_logit_list, total_attn_list)

    def evaluate_validation_metrics(self, model):
        """
        Evaluate the model's accuracy on the validation dataset and compute various metrics.
        
        Args:
        - model: The PyTorch model to evaluate.
        
        Returns:
        - A dictionary containing the computed metrics.
        """
        # Convert the model's output to probabilities using the sigmoid function.
        m = torch.nn.Sigmoid()

        # Create empty lists to hold the model's outputs and targets.
        output_list = []
        target_list = []

        # Evaluate the model on the validation dataset.
        model.eval()
        with torch.no_grad():
            for i, (images, target, _) in tqdm(enumerate(self.val_loader)):
                images = images.cuda()
                target = target.cuda()
                output = model(images)

                # Save the model's output and target for later analysis.
                output_list.append(m(output).detach().cpu().numpy())
                target_list.append(target.detach().cpu().numpy())

        # Compute various metrics on the model's outputs and targets.
        output = np.concatenate(output_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        if self.args.task == 'singlelabel':
            acc = utils.accuracyforsinglelabel(output, target)
        else:
            acc, = utils.accuracy(output, target)
        top_k = utils.multi_label_top_margin_k_accuracy(target, output, margin=0)
        mAP, _ = utils.compute_mean_average_precision(target, output)
        mROC, _ = utils.compute_mean_roc_auc(target, output)

        # Print the computed metrics.
        print(f'Validation: Acc: {acc:.4f}, Topk: {top_k:.4f}, mAP: {mAP:.4f}, mROC: {mROC:.4f}')

        # Return the computed metrics as a dictionary.
        result = {'acc': acc, 'top_k': top_k, 'mAP': mAP, 'mROC': mROC}
        return result
    
    def validate_model(self, model):
        """
        Validate the given model on the validation dataset and compute various metrics.
        
        Args:
        - model: The PyTorch model to validate.
        
        Returns:
        - A dictionary containing the computed metrics.
        """
        def isSameStateDict(dict1, dict2):
            for key in dict1.keys():
                if key not in dict2 or not torch.equal(dict1[key], dict2[key]):
                    return False
            return True
        
        if self.prev_model_state_dict is not None and isSameStateDict(model.state_dict(), self.prev_model_state_dict):
            print("Model is same as the previous model.")
            return self.prev_result

        result = self.evaluate_validation_metrics(model)

        # Save the model's state dictionary and metrics for future comparisons.
        if not isSameStateDict(model.state_dict(), self.prev_model_state_dict):
            self.prev_model_state_dict = copy.deepcopy(model.state_dict())
            self.prev_result = result
        
        return result
    
    def trainLocal(self, savename, modelid=0):
        epochs = self.args.initepochs
        model = self.localmodels[modelid]
        # tr_dataset = mydataset.data_cifar.Dataset_fromarray(self.private_data[modelid]['x'],self.private_data[modelid]['y'], train=True, verbose=False)
        # print('size of dataset', self.private_data[modelid]['x'].shape, self.private_data[modelid]['y'].shape)
        tr_dataset = mydataset.data_cifar.mydataset(self.private_data[modelid]['x'], self.private_data[modelid]['y'], train=True, verbose=False)
        train_loader = DataLoader(
            dataset=tr_dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.num_workers, sampler=None) 
        test_loader = self.val_loader
        args = self.args
        writer = self.writer
        datasize = len(tr_dataset)
        if self.args.task == 'singlelabel':
            criterion = nn.CrossEntropyLoss() #include softmax
        else:
            criterion = torch.nn.MultiLabelSoftMarginLoss() # torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
        m = torch.nn.Sigmoid()

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(epochs), eta_min=self.args.lrmin,)
        #
        bestacc = 0 
        bestname = ''
        for epoch in range(epochs):
            #train
            model.train()
            tracc = utils.AverageMeter()
            trloss = utils.AverageMeter()
            for i, (images, target, _) in enumerate(train_loader):
                # print(images.shape, target.shape)
                images = images.cuda()
                target = target.cuda()
                output = model(images)
                # import ipdb; ipdb.set_trace()
                loss = criterion(output, target)
                if self.args.task == 'singlelabel':
                    acc = utils.accuracyforsinglelabel(output.detach().cpu().numpy(), target.detach().cpu().numpy())
                else:
                    acc,  = utils.accuracy(m(output).detach().cpu().numpy(), target.detach().cpu().numpy())
                tracc.update(acc)
                trloss.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f'loss={trloss.avg:.2f}, acc={tracc.avg:.2f}')
            if writer is not None:
                writer.add_scalars(str(modelid)+'train', {'loss': trloss.avg}, epoch) #display_name=str(modelid)+'train'
                writer.add_scalars(str(modelid)+'train', {'acc': tracc.avg}, epoch) #display_name=str(modelid)+'train'
            #val
            model.eval()
            testacc = utils.AverageMeter()
            with torch.no_grad():
                for i, (images, target, _) in enumerate(test_loader):
                    # print(images.shape, target.shape)
                    images = images.cuda()
                    target = target.cuda()
                    output = model(images)
                    if self.args.task == 'singlelabel':
                        acc = utils.accuracyforsinglelabel(m(output).detach().cpu().numpy(), target.detach().cpu().numpy())
                    else:
                        acc, = utils.accuracy(m(output).detach().cpu().numpy(), target.detach().cpu().numpy())
                    testacc.update(acc)
                if writer is not None:
                    writer.add_scalar(str(modelid)+'testacc', testacc.avg, epoch, display_name=str(modelid)+'testacc')
                if testacc.avg > bestacc:
                    bestacc = testacc.avg
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestname = f'{savename[:-3]}_{(bestacc):.2f}.pt'
                    torch.save(model.state_dict(), bestname)
                    os.system(f'cp {bestname} {savename}')
                logging.info(f'{modelid}, Size={datasize}, Epoch={epoch}: testacc={testacc.avg}, Best======{bestacc}======')
            #
            scheduler.step()

        return bestacc