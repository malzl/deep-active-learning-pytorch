# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import numpy as np 
import torch
from statistics import mean
import gc
import os
import math
import sys
import time
import pickle
from copy import deepcopy
from tqdm import tqdm

from scipy.spatial import distance_matrix
import torch.nn as nn

from .vaal_util import train_vae_disc

class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy 

class CoreSetMIPSampling():
    """
    Implements coreset MIP sampling operation
    """
    def __init__(self, cfg, dataObj, isMIP = False):
        self.dataObj = dataObj
        self.cfg = cfg
        self.isMIP = isMIP

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        features = []
        
        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.type(torch.FloatTensor)
                temp_z, _ = clf_model(x)
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2, axis=1) + np.sum(X**2, axis=1).reshape((-1, 1))
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]
        uSetIds = n_lSet + np.arange(n_uSet)

        #order is important
        features = np.vstack((labeled, unlabeled))
        print("Started computing distance matrix of {}x{}".format(features.shape[0], features.shape[0]))
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        print("Distance matrix computed in {} seconds".format(end - start))
        greedy_indices = []
        for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE):
            if i != 0 and i % 500 == 0:
                print("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices), dtype=int)
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:], axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)
        
        remainSet = set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        remainSet = np.array(list(remainSet))

        return greedy_indices - n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = [None for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)]
        greedy_indices_counter = 0
        labeled = torch.from_numpy(labeled)
        unlabeled = torch.from_numpy(unlabeled)

        print(f"Labeled.shape: {labeled.shape}")
        print(f"Unlabeled.shape: {unlabeled.shape}")

        st = time.time()
        min_dist, _ = torch.min(self.gpu_compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), dim=0)
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
        print(f"time taken: {time.time() - st} seconds")

        temp_range = 500
        for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)
            
            min_dist = torch.cat((min_dist, torch.min(dist, dim=0)[0].reshape((1, min_dist.shape[1]))))

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        _, farthest = torch.max(min_dist, dim=1)
        greedy_indices[greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE - 1
        
        for i in tqdm(range(amount), desc="Constructing Active set"):
            dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            
            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))
            
            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices[greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        if self.isMIP:
            return greedy_indices, remainSet, math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def query(self, lSet, uSet, clf_model, dataset):
        assert clf_model.training == False, "Classification model expected in eval mode"
        assert clf_model.penultimate_active == True, "Classification model is expected in penultimate mode"    
        
        print("Extracting Lset Representations")
        lb_repr = self.get_representation(clf_model=clf_model, idx_set=lSet, dataset=dataset)
        print("Extracting Uset Representations")
        ul_repr = self.get_representation(clf_model=clf_model, idx_set=uSet, dataset=dataset)
        
        print("lb_repr.shape: ", lb_repr.shape)
        print("ul_repr.shape: ", ul_repr.shape)
        
        if self.isMIP == True:
            raise NotImplementedError
        else:
            print("Solving K Center Greedy Approach")
            start = time.time()
            greedy_indexes, remainSet = self.greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            end = time.time()
            print("Time taken to solve K center: {} seconds".format(end - start))
            activeSet = uSet[greedy_indexes]
            remainSet = uSet[remainSet]
        return activeSet, remainSet

class Sampling:
    """
    Here we implement different sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.dataObj = dataObj

    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1, 1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1, -1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def get_predictions(self, clf_model, idx_set, dataset):
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        tempIdxSetLoader.dataset.no_aug = True
        preds = []
        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
            with torch.no_grad():
                x = x.type(torch.FloatTensor)
                temp_pred = clf_model(x)
                temp_pred = torch.nn.functional.softmax(temp_pred, dim=1)
                preds.append(temp_pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tempIdxSetLoader.dataset.no_aug = False
        return preds

    def random(self, uSet, budgetSize):
        """
        Chooses <budgetSize> number of data points randomly from uSet.
        
        NOTE: The returned uSet is modified such that it does not contain active datapoints.

        INPUT
        ------
        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------
        Returns activeSet, uSet   
        """
        np.random.seed(self.cfg.RNG_SEED)

        assert isinstance(uSet, np.ndarray), "Expected uSet of type np.ndarray whereas provided is dtype:{}".format(type(uSet))
        assert isinstance(budgetSize, int), "Expected budgetSize of type int whereas provided is dtype:{}".format(type(budgetSize))
        assert budgetSize > 0, "Expected a positive budgetSize"
        assert budgetSize < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
            .format(len(uSet), budgetSize)

        tempIdx = [i for i in range(len(uSet))]
        np.random.shuffle(tempIdx)
        activeSet = uSet[tempIdx[0:budgetSize]]
        uSet = uSet[tempIdx[budgetSize:]]
        return activeSet, uSet

    def bald(self, budgetSize, uSet, clf_model, dataset):
        "Implements BALD acquisition function where we maximize information gain."
        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        uSetLoader.dataset.no_aug = True
        n_uPts = len(uSet)

        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS), desc="Dropout Iterations"):
            dropout_score = self.get_predictions(clf_model=clf_model, idx_set=uSet, dataset=dataset)
            score_All += dropout_score

            dropout_score_log = np.log2(dropout_score + 1e-6)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)
            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        F_X = Average_Entropy

        U_X = G_X - F_X
        sorted_idx = np.argsort(U_X)[::-1]
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        clf_model.train()
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def dbal(self, budgetSize, uSet, clf_model, dataset):
        """
        Implements deep bayesian active learning where uncertainty is measured by 
        maximizing entropy of predictions. This uncertainty method is choosen following
        the recent state of the art approach, VAAL. [SOURCE: Implementation Details in VAAL paper]
        
        In bayesian view, predictions are computed with the help of dropouts and 
        Monte Carlo approximation 
        """
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        uSetLoader.dataset.no_aug = True
        u_scores = []
        n_uPts = len(uSet)
        ptsProcessed = 0

        entropy_loss = EntropyLoss()

        print("len usetLoader: {}".format(len(uSetLoader)))
        temp_i = 0
        
        for k, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Feed Forward")):
            temp_i += 1
            x_u = x_u.type(torch.FloatTensor)
            z_op = np.zeros((x_u.shape[0], self.cfg.MODEL.NUM_CLASSES), dtype=float)
            for i in range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS):
                with torch.no_grad():
                    temp_op = clf_model(x_u)
                    temp_op = torch.nn.functional.softmax(temp_op, dim=1)
                    z_op = np.add(z_op, temp_op.cpu().numpy())

            z_op /= self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS
            z_op = torch.from_numpy(z_op)
            entropy_z_op = entropy_loss(z_op, applySoftMax=False)
            u_scores.append(entropy_z_op.numpy())
            ptsProcessed += x_u.shape[0]
            
        u_scores = np.concatenate(u_scores, axis=0)
        sorted_idx = np.argsort(u_scores)[::-1]
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def ensemble_var_R(self, budgetSize, uSet, clf_models, dataset):
        """
        Implements ensemble variance_ratio measured as the number of disagreement in committee 
        with respect to the predicted class. 
        If f_m is number of members agreeing to predicted class then 
        variance ratio(var_r) is evaludated as follows:
        
            var_r = 1 - (f_m / T); where T is number of commitee members

        For more details refer equation 4 in 
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf
        """
        from scipy import stats
        T = len(clf_models)

        for cmodel in clf_models:
            cmodel.eval()

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        uSetLoader.dataset.no_aug = True
        print("len usetLoader: {}".format(len(uSetLoader)))

        temp_i = 0
        var_r_scores = np.zeros((len(uSet), 1), dtype=float)
        
        for k, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Forward Passes through " + str(T) + " models")):
            x_u = x_u.type(torch.FloatTensor)
            ens_preds = np.zeros((x_u.shape[0], T), dtype=float)
            for i in range(len(clf_models)):
                with torch.no_grad():
                    temp_op = clf_models[i](x_u)
                    _, temp_pred = torch.max(temp_op, 1)
                    temp_pred = temp_pred.numpy()
                    ens_preds[:, i] = temp_pred
            _, mode_cnt = stats.mode(ens_preds, 1)
            temp_varr = 1.0 - (mode_cnt / T * 1.0)
            var_r_scores[temp_i:temp_i + x_u.shape[0]] = temp_varr

            temp_i = temp_i + x_u.shape[0]

        var_r_scores = np.squeeze(np.array(var_r_scores))
        print("var_r_scores: ")
        print(var_r_scores.shape)

        sorted_idx = np.argsort(var_r_scores)[::-1]
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def uncertainty(self, budgetSize, lSet, uSet, model, dataset):
        """
        Implements the uncertainty principle as an acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)
        
        clf = model
        
        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
                temp_u_rank = 1 - temp_u_rank
                u_ranks.append(temp_u_rank.detach().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        print(f"u_ranks.shape: {u_ranks.shape}")
        sorted_idx = np.argsort(u_ranks)[::-1]
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def entropy(self, budgetSize, lSet, uSet, model, dataset):
        """
        Implements the uncertainty principle as an acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model

        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank = temp_u_rank * torch.log2(temp_u_rank)
                temp_u_rank = -1 * torch.sum(temp_u_rank, dim=1)
                u_ranks.append(temp_u_rank.detach().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        print(f"u_ranks.shape: {u_ranks.shape}")
        sorted_idx = np.argsort(u_ranks)[::-1]
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def margin(self, budgetSize, lSet, uSet, model, dataset):
        """
        Implements the uncertainty principle as an acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model

        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]
                difference = -1 * difference 
                u_ranks.append(difference.detach().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        print(f"u_ranks.shape: {u_ranks.shape}")
        sorted_idx = np.argsort(u_ranks)[::-1]
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

class AdversarySampler:
    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.dataObj = dataObj
        self.budget = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        if cfg.DATASET.NAME == 'TINYIMAGENET':
            cfg.VAAL.Z_DIM = 64
            cfg.VAAL.IM_SIZE = 64
        else:
            cfg.VAAL.Z_DIM = 32
            cfg.VAAL.IM_SIZE = 32

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
        return dists

    def vaal_perform_training(self, lSet, uSet, dataset, debug=False):
        oldmode = self.dataObj.eval_mode
        self.dataObj.eval_mode = True
        self.dataObj.eval_mode = oldmode

        vae, disc = train_vae_disc(self.cfg, lSet, uSet, dataset, self.dataObj, debug)
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset)

        vae.eval()
        disc.eval()

        return vae, disc, uSetLoader

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = []
    
        min_dist = np.min(self.compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        temp_range = 1000
        for j in range(1, labeled.shape[0], temp_range):
            if j + temp_range < labeled.shape[0]:
                dist = self.compute_dists(labeled[j:j + temp_range, :], unlabeled)
            else:
                dist = self.compute_dists(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE - 1
        for i in range(amount):
            if i != 0 and i % 500 == 0:
                print("{} Sampled out of {}".format(i, amount + 1))
            dist = self.compute_dists(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        return greedy_indices, remainSet

    def get_vae_activations(self, vae, dataLoader):
        acts = []
        vae.eval()
        
        temp_max_iter = len(dataLoader)
        print("len(dataloader): {}".format(temp_max_iter))
        temp_iter = 0
        for x, y in dataLoader:
            x = x.type(torch.FloatTensor)
            _, _, mu, _ = vae(x)
            acts.append(mu.cpu().numpy())
            if temp_iter % 100 == 0:
                print(f"Iteration [{temp_iter}/{temp_max_iter}] Done!!")

            temp_iter += 1
        
        acts = np.concatenate(acts, axis=0)
        return acts

    def get_predictions(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        assert vae.training == False, "Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images, _ in data:
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds = all_preds.cpu().numpy()
        return all_preds

    def vae_sample_for_labeling(self, vae, uSet, lSet, unlabeled_dataloader, lSetLoader):
        vae.eval()
        print("Computing activations for uset....")
        u_scores = self.get_vae_activations(vae, unlabeled_dataloader)
        print("Computing activations for lset....")
        l_scores = self.get_vae_activations(vae, lSetLoader)
        
        print("l_scores.shape: ", l_scores.shape)
        print("u_scores.shape: ", u_scores.shape)
        
        dist_matrix = self.efficient_compute_dists(l_scores, u_scores)
        print("Dist_matrix.shape: ", dist_matrix.shape)

        min_scores = np.min(dist_matrix, axis=1)
        sorted_idx = np.argsort(min_scores)[::-1]

        activeSet = uSet[sorted_idx[0:self.budget]]
        remainSet = uSet[sorted_idx[self.budget:]]

        return activeSet, remainSet

    def sample_vaal_plus(self, vae, disc_task, data):
        all_preds = []
        all_indices = []

        assert vae.training == False, "Expected vae model to be in eval mode"
        assert disc_task.training == False, "Expected disc_task model to be in eval mode"

        temp_idx = 0
        for images, _ in data:
            images = images.type(torch.FloatTensor)
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds, _ = disc_task(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds *= -1

        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices), "Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet

    def sample(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        assert vae.training == False, "Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images, _ in data:
            images = images.type(torch.FloatTensor)
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds *= -1

        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices), "Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet

    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, uSet):
        """
        Picks samples from uSet to form activeSet.

        INPUT
        ------
        vae: object of model VAE

        discriminator: object of model discriminator

        unlabeled_dataloader: Sequential dataloader iterating over uSet

        uSet: Collection of unlabelled datapoints

        NOTE: Please pass the unlabelled dataloader as sequential dataloader else the
        results won't be appropriate.

        OUTPUT
        -------

        Returns activeSet, [remaining]uSet
        """
        unlabeled_dataloader.dataset.no_aug = True
        activeSet, remainSet = self.sample(vae, discriminator, unlabeled_dataloader)

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        unlabeled_dataloader.dataset.no_aug = False
        return activeSet, remainSet
