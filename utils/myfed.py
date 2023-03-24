import os
import numpy as np
import logging
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import utils.utils as utils
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
        self.prev_model_state_dist = {}
        
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
            self.sub_criterion = Loss.MHALoss()
        else:
            self.sub_criterion = None
        
        # distill optimizer
        self.bestacc = 0
        self.best_statdict = self.central.state_dict()
        
        # import ipdb; ipdb.set_trace()
        #path to save ckpt
        self.rootdir = f'./checkpoints/{args.dataset}/{args.model_name}_{args.task}/a{self.args.alpha}+sd{self.args.seed}+e{self.args.initepochs}+b{self.args.batchsize}+l{self.args.lossmode}+sl{self.args.sublossmode}'
        if not os.path.isdir(self.rootdir):
            os.makedirs(self.rootdir)
        if initpth:
            if not args.subpath:
                if args.oneshot:
                    args.subpath = f'oneshot_c{args.C}_q{args.quantify}_n{args.noisescale}'
                else:
                    args.subpath = f'oneshot_c{args.C}_q{args.quantify}_n{args.noisescale}'
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
                    utils.load_dict(savename, self.localmodels[n])
                    # acc = self.validate_model(self.localmodels[n])
                    acc = 0.0
                else:
                    logging.info(f'Init Local{n}, Epoch={epochs}......')
                    acc = self.trainLocal(savename, modelid=n)
                logging.info(f'Init Local{n}--Epoch={epochs}, Acc:{(acc):.2f}')
     
    def init_central(self):
        initname = os.path.join(self.rootdir, self.args.initcentral)
        if os.path.exists(initname):
            utils.load_dict(initname, self.central)
            acc = self.validate_model(self.central)
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
            total_logits = self.central(images).detach()
        else:
            total_logits = []
            for n in selectN:
                tmodel = copy.deepcopy(self.localmodels[n])
                logits = torch.sigmoid(tmodel(images).detach())
                total_logits.append(logits)
                del tmodel
            total_logits = torch.stack(total_logits)
        return total_logits
        
    def compute_class_weights(self, class_counts):
        """
        Args:
            class_counts (torch.Tensor): (num_samples, num_classes)
        Returns:
            class_weights (torch.Tensor): (num_samples, num_classes)
        """
        # Normalize the class counts per sample
        class_weights = class_counts / class_counts.sum(dim=1, keepdim=True)
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

    def get_ensemble_logits(self, total_logits, selectN):
        class_counts = self.countN[selectN]
        class_weights = self.compute_class_weights(torch.from_numpy(class_counts).float().cuda())
        if self.voteout:
            ensemble_logits = Loss.weight_psedolabel(total_logits, self.countN[selectN], noweight=True).detach()
        else:
            ensemble_logits = self.compute_ensemble_logits(total_logits, class_weights)
        return ensemble_logits

    def compute_euclidean_norm(self, vector_a, vector_b):
        return torch.tensor(1) - torch.sqrt(torch.sum((vector_a - vector_b) ** 2, dim=-1))

    def compute_cosine_similarity(self, vector_a, vector_b):
        # print(vector_a.shape, vector_b.shape)
        return torch.tensor(1) - (torch.sum(vector_a * vector_b, dim=-1) / (torch.norm(vector_a, dim=-1) * torch.norm(vector_b, dim=-1)))

    def calculate_normalized_similarity_weights(self, target_vectors, client_vectors, similarity_method='euclidean'):
        if similarity_method == 'euclidean':
            similarity_function = self.compute_euclidean_distance
        elif similarity_method == 'cosine':
            similarity_function = self.compute_cosine_similarity
        else:
            raise ValueError("Invalid similarity method. Choose 'euclidean' or 'cosine'.")

        target_vectors_expanded = target_vectors.unsqueeze(0)  # Shape: (1, batch_size, n_class)
        
        similarities = similarity_function(target_vectors_expanded, client_vectors)  # Shape: (n_client, batch_size)
        # print(similarities.shape)
        mean_similarities = torch.mean(similarities, dim=1)  # Shape: (n_client)
        # print(mean_similarities)
        normalized_similarity_weights = mean_similarities / torch.sum(mean_similarities)  # Shape: (n_client)
        # print(normalized_similarity_weights)
        return normalized_similarity_weights


    def distill_onemodel_batch(self, model, images, selectN, optimizer, usecentral=True):
        ''' for ensemble logits '''
        total_logits = self.get_total_logits(model, images, selectN, usecentral=usecentral)
        ensemble_logits = self.get_ensemble_logits(total_logits, selectN)
        sim_weights = self.calculate_normalized_similarity_weights(ensemble_logits, total_logits, "cosine")
        
        # ''' for subloss '''
        # if self.args.sublossmode=='at':
        # if self.args.sublossmode=='mha':
        #     # union, intersection = Loss.weight_multihead_attention_map(total_mha, self.countN)

        model.train()
        central_logits = torch.sigmoid(model(images))
        logit_loss = self.criterion(central_logits, ensemble_logits)
        # ensemble_logits = nclient * batch * ncls
        
        if self.args.sublossmode=='at':
            total_gradcam = self.getTotalGradcam(model, images, selectN, usecentral=usecentral)
            union, intersection = Loss.weight_gradcam(total_gradcam, self.countN)
            gradcam = model.module.get_class_activation_map(images, y=None)
            sub_loss = self.sub_criterion(union, intersection, gradcam)
        elif self.args.sublossmode=='mha':
            total_mha = self.get_total_multi_head_attention_map(model, images, selectN)
            mha = model.module.get_attention_maps_postprocessing(images)
            sub_loss = self.sub_criterion(total_mha, mha, sim_weights)
        else : 
            sub_loss = torch.tensor(0.0).cuda()
        # print('loss', logit_loss, 'sub_loss', sub_loss, 'total_loss', logit_loss+sub_loss)
        # total_loss = logit_loss + sub_loss
        # print('logit_loss : ', logit_loss, 'sub_loss : ', sub_loss, 'total_loss : ', total_loss)
        optimizer[0].zero_grad()
        # total_loss.backward()
        print('sub_loss : ', sub_loss.grad_fn)
        sub_loss.backward()
        optimizer[0].step()
        return logit_loss, sub_loss

    def distill_local_central(self):
        args = self.args
        net = self.central
        last_layer_name = list(net.module.named_children())[-1][0]
        parameters = [
            {'params': [p for n, p in net.module.named_parameters() if last_layer_name not in n], 'lr': args.dis_lr},
            {'params': [p for n, p in net.module.named_parameters() if last_layer_name in n], 'lr': args.dis_lr*100},
        ]
        if self.args.optim == 'SGD':
            optimizer = optim.SGD( params= parameters, lr=self.args.dis_lr, momentum=self.args.momentum, weight_decay=self.args.wdecay)
            optimizer2 = optim.SGD( params= net.module.parameters(), lr=self.args.dis_lr*1000, momentum=self.args.momentum, weight_decay=self.args.wdecay)
        else:    
            optimizer = optim.Adam( params= parameters, lr=self.args.dis_lr, betas=(self.args.momentum, 0.999), weight_decay=args.wdecay)
            optimizer2 = optim.Adam( params= net.module.parameters(), lr=self.args.dis_lr*1000, betas=(self.args.momentum, 0.999), weight_decay=args.wdecay)
            
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
            for i, (images, _, _) in tqdm(enumerate(self.distil_loader)):
                images = images.cuda()
                if self.args.C<1:
                    selectN = random.sample(self.locallist, int(args.C*self.N_parties))
                
                loss, subloss = self.distill_onemodel_batch(self.central, images, selectN, [optimizer, optimizer2], usecentral=False) 
                loss_avg += loss.item()
                sub_loss_avg += subloss.item()
                step += 1
            
            loss_avg /= len(self.distil_loader)
            sub_loss_avg /= len(self.distil_loader)
            
            if epoch%1==0:
                last_acc = self.validate_model(self.central)
                acc = last_acc
                 
            if self.writer is not None:
                self.writer.add_scalar('loss', loss, epoch, display_name='loss')
                self.writer.add_scalar('at_loss', subloss, epoch, display_name='at_loss')
                self.writer.add_scalar('DisACC', acc, epoch, display_name='DisACC')
            if self.experiment is not None:
                self.experiment.log_metric('lg_loss', loss_avg, step=epoch)
                self.experiment.log_metric('at_loss', sub_loss_avg, step=epoch)
                self.experiment.log_metric('DisACC', acc, step=epoch)
            
            logging.info(f'Epoch{epoch}, Acc{(acc):.2f}')
            stop, best = earlyStop.update(step, acc)
            if best:
                if bestname:
                    os.system(f'rm {bestname}')
                bestacc = acc
                bestname = f'{savename}_e{epoch}_{(bestacc):.2f}.pt'
                torch.save(self.central.state_dict(), bestname)
                logging.info(f'========Best...Epoch{epoch}, Acc{(acc):.2f}')
            else:
                logging.info(f'Epoch{epoch}, Acc{(acc):.2f}')
                
            scheduler.step()
            if stop:
                logging.info(f'Early Stop at Epoch{epoch}, Acc{(acc):.2f}')
                break
    
    def validate_model(self, model):
        model.eval()
        # testacc = utils.AverageMeter()
        m = torch.nn.Sigmoid()
        output_list = []
        target_list = []
        def isSameDict(d1, d2):
            for k in d1:
                if k not in d2:
                    return False
                if not torch.equal(d1[k], d2[k]):
                    return False
            return True
        if isSameDict(model.state_dict(), self.prev_model_state_dist):
            return self.prev_acc
        
        with torch.no_grad():
            for i, (images, target, _) in enumerate(self.val_loader):
                images = images.cuda()
                target = target.cuda()
                output = model(images)
                
                output_list.append(m(output).detach().cpu().numpy())
                target_list.append(target.detach().cpu().numpy())
                
                # testacc.update(acc)
        output = np.concatenate(output_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        acc, = utils.accuracy(output, target)
        self.prev_acc = acc
        self.prev_model_state_dist = copy.deepcopy(model.state_dict())
        return acc
    
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
        # criterion = nn.CrossEntropyLoss() #include softmax
        # multi label
        criterion = torch.nn.MultiLabelSoftMarginLoss() # torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=3e-4)
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