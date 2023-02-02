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

class FedMAD:
    def __init__(self, central, distil_loader, private_data, val_loader, 
                   writer, args, initpth=True, localmodel=None):
        # import ipdb; ipdb.set_trace()
        self.localmodel = localmodel
        self.central = central
        self.N_parties = args.N_parties
        self.distil_loader = distil_loader
        self.private_data = private_data
        self.val_loader = val_loader
        self.writer = writer
        self.args = args
        
        self.prevacc = 0
        self.prev_model_state_dist = {}
        
        # ensemble and loss
        totalN = []
        for n in range(self.N_parties):
            totalN.append(private_data[n]['x'].shape[0])
        totalN = torch.tensor(totalN) #nlocal
        # assert totalN.sum() == 50000
        print('totalN', totalN)
        self.totalN = totalN.cuda()#nlocal*1*1
        countN = np.zeros((self.N_parties, self.args.N_class))
        for n in range(self.N_parties):
            counterclass = np.sum(private_data[n]['y'], axis=0)
            for m in range(self.args.N_class):
                # print('n : ', n, 'm : ', m, 'private_data[n][y] : ', private_data[n]['y'])
                # countN[n][m] = (self.private_data[n]['y'] == m).sum()
                countN[n][m] = counterclass[m]
                
        # assert countN.sum() == 50000
        print('countN', countN)
        self.countN = torch.tensor(countN).cuda()      
        self.locallist= list(range(0, self.N_parties))# local number list
        self.clscnt = args.clscnt# if ensemble is weighted by number of local sample
        self.voteout = args.voteout
        if args.lossmode == 'l1':
            self.criterion = torch.nn.L1Loss(reduce=True)
        elif args.lossmode=='kl': 
            self.criterion = Loss.kl_loss(T=3, singlelabel=True)
        else:
            raise ValueError
            
        if args.sublossmode=='at':
            self.sub_criterion = Loss.at_loss()
        elif args.sublossmode=='mha':
            self.sub_criterion = Loss.mha_loss()
        else:
            self.sub_criterion = None
        
        # distill optimizer
        self.bestacc = 0
        self.best_statdict = self.central.state_dict()
        
        # import ipdb; ipdb.set_trace()
        #path to save ckpt
        self.rootdir = f'./checkpoints/{args.dataset}/a{self.args.alpha}+sd{self.args.seed}+e{self.args.initepochs}+b{self.args.batchsize}+l{self.args.lossmode}+sl{self.args.sublossmode}'
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

    def init_locals(self, initpth=True, init_dir=''):
        epochs = self.args.initepochs
        if self.localmodel is None:
            self.localmodels = utils.copy_parties(self.N_parties, self.central)
        else:
            self.localmodels = utils.copy_parties(self.N_parties, self.localmodel)
        if not init_dir:
            init_dir = self.rootdir
        if initpth:
            for n in range(self.N_parties):
                # savename = os.path.join(init_dir, str(n)+'.pt')
                savename = os.path.join(init_dir, f'model-{n}.pth')
                if os.path.exists(savename):
                    #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
                    logging.info(f'Loading Local{n}......')
                    print('filepath : ', savename)
                    utils.load_dict(savename, self.localmodels[n])
                    acc = self.validate_model(self.localmodels[n])
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
            # del tmodel
        return total_gradcam

    def getTotalMultiHeadAttentionMap(self, model, images, selectN, usecentral=True):
        total_mha = [] 
        for n in selectN:
            tmodel = copy.deepcopy(self.localmodels[n])
            mha, th_attn = tmodel.module.get_attention_maps_postprocessing(images)
            total_mha.append(mha) # batch * nhead * 224 * 224
            # del tmodel
        return total_mha # n_client * batch * nhead * 224 * 224

    def getEnsemble_logits(self, model, images, selectN, localweight, usecentral=True):
        if usecentral:
            ensemble_logits = self.central(images).detach()
        else:
            #get local
            total_logits = []
            m = torch.nn.Sigmoid()
            for n in selectN:
                tmodel = copy.deepcopy(self.localmodels[n])
                logits = tmodel(images).detach()
                total_logits.append(m(logits))
                # del tmodel
            total_logits = torch.stack(total_logits)
            if self.voteout:
                ensemble_logits = Loss.weight_psedolabel(total_logits, self.countN[selectN], noweight=True).detach()
            else:
                ensemble_logits = (total_logits*localweight).detach().sum(dim=0) #batch*ncls
        return ensemble_logits

    def distill_onemodel_batch(self, model, images, selectN, localweight, optimizer, usecentral=True):
        ''' for ensemble logits '''
        ensemble_logits = self.getEnsemble_logits(model, images, selectN, localweight, usecentral=usecentral)

        ''' for subloss '''
        if self.args.sublossmode=='at':
            total_gradcam = self.getTotalGradcam(model, images, selectN, usecentral=usecentral)
            union, intersection = Loss.weight_gradcam(total_gradcam, self.countN)
        if self.args.sublossmode=='mha':
            total_mha = self.getTotalMultiHeadAttentionMap(model, images, selectN, usecentral=usecentral)
            union, intersection = Loss.weight_multihead_attention_map(total_mha, self.countN)

        model.train()
        m = torch.nn.Sigmoid()
        output = model(images)
        logit_loss = self.criterion(m(output), ensemble_logits)

        if self.args.sublossmode=='at':
            gradcam = model.module.get_class_activation_map(images, y=None)
            sub_loss = self.sub_criterion(union, intersection, gradcam)
        elif self.args.sublossmode=='mha':
            mha, th_attn = model.module.get_attention_maps_postprocessing(images)
            sub_loss = self.sub_criterion(union, intersection, mha)
        else : 
            sub_loss = torch.tensor(0.0).cuda()
        print('loss', logit_loss, 'sub_loss', sub_loss, 'total_loss', logit_loss+sub_loss)
        loss = logit_loss + sub_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return logit_loss, sub_loss

    def distill_local_central(self):
        args = self.args
        if self.args.optim == 'SGD':
            optimizer = optim.SGD(
                self.central.parameters(), lr=self.args.dis_lr, momentum=args.momentum, weight_decay=args.wdecay)
        else:    
            optimizer = optim.Adam(
                self.central.parameters(), lr=self.args.dis_lr, momentum=args.momentum, weight_decay=args.wdecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.fedrounds), eta_min=args.dis_lrmin,)
        
        savename = os.path.join(self.savedir, f'q{args.quantify}_n{args.noisescale}_{args.optim}_b{args.disbatchsize}_{args.dis_lr}_{args.fedrounds}_{args.dis_lrmin}_m{args.momentum}')
        bestacc = self.bestacc
        bestname = ''
        selectN = self.locallist
        if self.clscnt:
            countN = self.countN
            localweight = countN/countN.sum(dim=0)
            localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
        else:
            localweight = 1.0*self.totalN/self.totalN.sum()
            localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
        
        step = 0
        for epoch in range(0, args.fedrounds):
            for i, (images, _, _) in enumerate(self.distil_loader):
                images = images.cuda()
                countN = self.countN
                if self.args.C<1:
                    selectN = random.sample(self.locallist, int(args.C*self.N_parties))
                    #selectN = self.locallist[select]
                    countN = self.countN[selectN]
                    if self.clscnt:
                        localweight = countN/countN.sum(dim=0)#nlocal*nclass
                        localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
                    else:
                        totalN = self.totalN[selectN]
                        localweight = 1.0*totalN/totalN.sum()
                        localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
                
                loss, subloss = self.distill_onemodel_batch(self.central, images, selectN, localweight, optimizer, usecentral=False) 
                step += 1
                acc = self.validate_model(self.central)
                if self.writer is not None:
                    self.writer.add_scalar('loss', loss.item(), step, display_name='loss')
                    self.writer.add_scalar('at_loss', subloss.item(), step, display_name='at_loss')
                    self.writer.add_scalar('DisACC', acc, step, display_name='DisACC')
                if acc>bestacc:
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestacc = acc
                    bestname = f'{savename}_i{step}_{(bestacc):.2f}.pt'
                    torch.save(self.central.state_dict(), bestname)
                    logging.info(f'========Best...Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
            scheduler.step()
            logging.info(f'Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
      
    def distill_local_central_joint(self):
        if self.args.initcentral:
            self.init_central()
            usecentral = True
        else:
            usecentral = False
        if self.args.C==1:
            selectN = self.locallist
            countN = self.countN
            localweight = countN/countN.sum(dim=0)
            localweight = localweight.unsqueeze(dim=1)
        #optimizer
        self.totalSteps = int(self.args.steps_round*128/(self.args.disbatchsize))
        self.earlystopSteps = int(self.totalSteps/5)
        self.max_epochs_round = 1+int(self.totalSteps/len(self.distil_loader))

        self.localsteps = np.zeros(self.N_parties)
        local_optimizers = []
        local_schedulers = []
        for n in range(self.N_parties):
            if self.args.optim == 'SGD':
                optimizer = optim.SGD(
                    self.localmodels[n].parameters(), lr=self.args.dis_lr, momentum=self.args.momentum,  weight_decay=self.args.wdecay)
            else:    
                optimizer = optim.Adam(
                    self.localmodels[n].parameters(), lr=self.args.dis_lr,  betas=(self.args.momentum, 0.999), weight_decay=self.args.wdecay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.args.fedrounds), eta_min=self.args.dis_lrmin,)
            local_optimizers.append(optimizer)
            local_schedulers.append(scheduler)

        for roundd in range(1, 1+self.args.fedrounds):
            if self.args.C<1:    
                selectN = random.sample(self.locallist, int(self.args.C*self.N_parties))
                countN = self.countN[selectN]
                localweight = countN/countN.sum(dim=0)
                localweight = localweight.unsqueeze(dim=1)
            acc = self.validateLocalTeacher(selectN, localweight)
            logging.info(f'*****************Round{roundd},LocalAVG:{(acc):.2f}***********************')
            self.updateLocals(local_optimizers, roundd, selectN, localweight, usecentral=usecentral)
            # import ipdb; ipdb.set_trace()
            for n in selectN:
                local_schedulers[n].step()
            acc = self.validateLocalTeacher(selectN, localweight)
            logging.info(f'*****************Round{roundd},LocalAVG:{(acc):.2f}***********************')
            self.updateCentral(roundd, selectN, localweight)
            
    def updateLocals(self, optimizers, roundd, selectN, selectweight, usecentral=False): #only update for the selected
        for n in selectN:
            logging.info(f'---------------------Local-{n}------------------------')
            self.distill_onelocal(roundd, n, optimizers[n],  selectN, selectweight, usecentral=usecentral, writer=self.writer)
            
    def updateCentral(self, roundd, selectN, selectweight):
        #
        step = 0
        args = self.args
        earlyStop = utils.EarlyStop(self.earlystopSteps, self.totalSteps, bestacc=self.bestacc)
        self.best_statdict = copy.deepcopy(self.central.state_dict())
        
        if self.args.optim == 'SGD':
            optimizer = optim.SGD(
                self.central.parameters(), lr=self.args.dis_lr, momentum=self.args.momentum, weight_decay=args.wdecay)
        else:    
            optimizer = optim.Adam(
                self.central.parameters(), lr=self.args.dis_lr,  betas=(self.args.momentum, 0.99), weight_decay=args.wdecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.totalSteps), eta_min=args.dis_lrmin,)

        for epoch in range(self.max_epochs_round):
            for i, (images, _, _) in enumerate(self.distil_loader):
                images = images.cuda()
                loss, subloss = self.distill_onemodel_batch(self.central, images, selectN, selectweight, optimizer, usecentral=False)
                acc = self.validate_model(self.central)
                step += 1
                stop, best = earlyStop.update(step, acc)
                if best:
                    logging.info(f'Iter{step}, best for now:{acc}')
                    self.best_statdict = copy.deepcopy(self.central.state_dict())
                if stop:
                    break
                else:
                    logging.info(f'===R{roundd}, Epoch:{epoch}/{self.max_epochs_round}, acc{acc}, best{earlyStop.bestacc}')
                    scheduler.step()
                    continue
            break
        #
        self.central.load_state_dict(self.best_statdict, strict=True)
        savename = os.path.join(self.savedir, f'r{roundd}_{(earlyStop.bestacc):.2f}.pt')
        torch.save(self.best_statdict, savename)
        logging.info(f'==================Round{roundd},Init{(self.bestacc):.2f}, Acc{(earlyStop.bestacc):.2f}====================')
        self.bestacc = earlyStop.bestacc
        if self.writer is not None:
            self.writer.add_scalar('DisACC', self.bestacc, roundd) 
    
    def distill_onelocal(self, roundd, modelid, optimizer, selectN, localweight, usecentral = False, writer=None):
        model = self.localmodels[modelid]
        initacc = self.validate_model(model)
        bestdict = copy.deepcopy(model.state_dict())
        earlyStop = utils.EarlyStop(self.earlystopSteps, self.totalSteps, bestacc=initacc)
        #
        savename = os.path.join(self.savedir, f'Local{modelid}_r{roundd}')
        writermark = f'Local{modelid}'
        step = 0
        globalstep = self.localsteps[modelid]
        bestname = ''
        for epoch in range(0, self.max_epochs_round):
            for i, (images, _, _) in enumerate(self.distil_loader):
                images = images.cuda()
                loss, subloss = self.distill_onemodel_batch(model, images, selectN, localweight, optimizer, usecentral=usecentral)
                step += 1
                globalstep += 1
                acc = self.validate_model(model)
                if writer is not None:
                    writer.add_scalar(writermark, acc, globalstep)
                stop, best = earlyStop.update(step, acc)
                if best:
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestacc = acc
                    bestdict = copy.deepcopy(model.state_dict())
                    bestname = f'{savename}_i{int(globalstep):d}_{(bestacc):.2f}.pt'
                    torch.save(model.state_dict(), bestname)
                    logging.info(f'========Best...Iter{globalstep},Epoch{epoch}, Acc{(acc):.2f}')
                if stop:
                    break
            else:
                logging.info(f'Epoch:{epoch}/{self.max_epochs_round}, acc{acc}, best{earlyStop.bestacc}')
                #scheduler.step()
                continue
            break
        lastacc = self.validate_model(model)        
        model.load_state_dict(bestdict, strict=True)
        acc = self.validate_model(model)
        logging.info(f'R{roundd}, L{modelid}, ========Init{(initacc):.2f}====Final{(lastacc):.2f}====Best{(earlyStop.bestacc):.2f}')
        self.localsteps[modelid] = globalstep
    
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

    def validateLocalTeacher(self, selectN, localweight):   
        testacc = utils.AverageMeter()
        with torch.no_grad():
            for i, (images, target, _) in enumerate(self.val_loader):
                logits = []
                images = images.cuda()
                target = target.cuda()
                for n in selectN:
                    output = self.localmodels[n](images).detach()
                    logits.append(output)
                logits = torch.stack(logits)
                if self.voteout:
                    ensemble_logits = Loss.weight_psedolabel(logits, self.countN[selectN], noweight=True)
                else:
                    ensemble_logits = (logits*localweight).sum(dim=0)
                acc, = utils.accuracy(ensemble_logits, target)
                testacc.update(acc)
        return testacc.avg
    
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
        criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean') # torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=3e-4)
        m = torch.nn.Sigmoid()

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(epochs), eta_min=args.lrmin,)
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
                loss = criterion(m(output), target)
                acc,  = utils.accuracy(m(output), target)
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
                    acc, = utils.accuracy(m(output), target)
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
       
    def update_distill_loader_wlocals(self, public_dataset):
        """
        save local prediction for one-shot distillation
        """
        total_logits = []
        m = torch.nn.Sigmoid()
        for i, (images, _, idx) in enumerate(self.distil_loader):
            # import ipdb; ipdb.set_trace()
            images = images.cuda()
            batch_logits = []
            for n in self.locallist:
                tmodel = copy.deepcopy(self.localmodels[n])
                logits = tmodel(images).detach()
                batch_logits.append(m(logits))
                # del tmodel
            batch_logits = torch.stack(batch_logits).cpu()#(nl, nb, ncls)
            total_logits.append(batch_logits)
        self.total_logits = torch.cat(total_logits,dim=1).permute(1,0,2) #(nsample, nl, ncls)
        if self.args.dataset=='cifar10':
            assert public_dataset.aug == False
            public_dataset.aug = True
        if self.args.dataset=='pascal_voc2012':
            assert public_dataset.aug == False
            public_dataset.aug = True
        
        self.distil_loader = DataLoader(
            dataset=public_dataset, batch_size=self.args.disbatchsize, shuffle=True, 
            num_workers=self.args.num_workers, pin_memory=True, sampler=None)

    def distill_batch_oneshot(self, model, images, idx, selectN, localweight, optimizer):
        total_logits = self.total_logits[idx].permute(1,0,2) #nlocal*batch*ncls
        total_logits = total_logits[torch.tensor(selectN)].to(images.device) #nlocal*batch*ncls
        #if quantify

        if self.voteout:
            ensemble_logits, votemask = Loss.weight_psedolabel(total_logits, self.countN[selectN])
        else:    
            ensemble_logits = (total_logits*localweight).sum(dim=0) #batch*ncls
        #if noise

        model.train()
        central_logits = model(images)
        m = torch.nn.Sigmoid()
        loss = self.criterion(m(central_logits), ensemble_logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
        
    def distill_local_central_oneshot(self):
        args = self.args
        if self.args.optim == 'SGD':
            optimizer = optim.SGD(
                self.central.parameters(), lr=self.args.dis_lr, momentum=args.momentum, weight_decay=args.wdecay)
        else:    
            optimizer = optim.Adam( 
                self.central.parameters(), lr=self.args.dis_lr, weight_decay=args.wdecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.fedrounds), eta_min=args.dis_lrmin,)
        
        savename = os.path.join(self.savedir, f'osp{args.public_percent}_q{args.quantify}_n{args.noisescale}_{args.optim}_b{args.disbatchsize}_{args.dis_lr}_{args.fedrounds}_{args.dis_lrmin}_m{args.momentum}')
        bestacc = self.bestacc
        bestname = ''
        selectN = self.locallist
        if self.clscnt:
            countN = self.countN
            localweight = countN/countN.sum(dim=0)
            localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
        else:
            localweight = 1.0*self.totalN/self.totalN.sum()
            localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
        
        step = 0
        for epoch in range(0, args.fedrounds):
            for i, (images, _, idx) in enumerate(self.distil_loader):
                images = images.cuda()
                countN = self.countN
                if self.args.C<1:
                    selectN = random.sample(self.locallist, int(args.C*self.N_parties))
                    countN = self.countN[selectN]
                    if self.clscnt:
                        localweight = countN/countN.sum(dim=0)#nlocal*nclass
                        localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
                    else:
                        totalN = self.totalN[selectN]
                        localweight = 1.0*totalN/totalN.sum()
                        localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
                
                loss = self.distill_batch_oneshot(self.central, images, idx, selectN, localweight, optimizer)
                step += 1
                acc = self.validate_model(self.central)
                if self.writer is not None:
                    self.writer.add_scalar('loss', loss.item(), step)
                    self.writer.add_scalar('DisACC', acc, step)
                if acc>bestacc:
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestacc = acc
                    bestname = f'{savename}_i{step}_{(bestacc):.2f}.pt'
                    torch.save(self.central.state_dict(), bestname)
                    logging.info(f'========Best...Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
            scheduler.step()
            logging.info(f'Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
