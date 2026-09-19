"""Traveling-model and FedAvg simulation for patient-disjoint MRI sites."""
from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn import BrainMRICNN
from src.utils.metrics import classification_report_dict


def _local_train(model, loader, device, local_steps, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    it = iter(loader)
    for _ in range(local_steps):
        try:
            img, y, _mask = next(it)
        except StopIteration:
            it = iter(loader)
            img, y, _mask = next(it)
        img, y = img.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(img), y)
        loss.backward()
        opt.step()
    return model


def _average_state_dicts(state_dicts: List[dict]) -> dict:
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        if avg[key].dtype.is_floating_point:
            avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
    return avg


def _evaluate(model, loader, device, threshold=0.5):
    model.to(device).eval()
    probs, labels = [], []
    with torch.no_grad():
        for img, y, _mask in loader:
            p = torch.softmax(model(img.to(device)), dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return classification_report_dict(labels, probs, threshold=threshold), probs, labels


def run_traveling_model(site_loaders: List[DataLoader], test_loader, device,
                         rounds: int, local_steps: int, lr=1e-3, seed=0,
                         in_channels=1):
    torch.manual_seed(seed)
    model = BrainMRICNN(in_channels=in_channels).to(device)
    history = []
    for rnd in range(rounds):
        for loader in site_loaders:
            _local_train(model, loader, device, local_steps, lr=lr)
        report, _, _ = _evaluate(model, test_loader, device)
        history.append({"round": rnd, **report})
    return model, history


def run_fedavg(site_loaders: List[DataLoader], test_loader, device,
                rounds: int, local_steps: int, lr=1e-3, seed=0,
                in_channels=1):
    torch.manual_seed(seed)
    global_model = BrainMRICNN(in_channels=in_channels).to(device)
    history = []
    for rnd in range(rounds):
        local_states = []
        for loader in site_loaders:
            local_model = copy.deepcopy(global_model)
            _local_train(local_model, loader, device, local_steps, lr=lr)
            local_states.append(local_model.state_dict())
            del local_model
        global_model.load_state_dict(_average_state_dicts(local_states))
        report, _, _ = _evaluate(global_model, test_loader, device)
        history.append({"round": rnd, **report})
    return global_model, history


def compare_strategies(site_loaders, test_loader, device, rounds=6, local_steps=30,
                        n_repeats=3, lr=1e-3, in_channels=1):
    travel_final_auprc, fedavg_final_auprc = [], []
    travel_histories, fedavg_histories = [], []
    for rep in range(n_repeats):
        _, hist_t = run_traveling_model(
            site_loaders, test_loader, device, rounds, local_steps,
            lr=lr, seed=100 + rep, in_channels=in_channels,
        )
        _, hist_f = run_fedavg(
            site_loaders, test_loader, device, rounds, local_steps,
            lr=lr, seed=100 + rep, in_channels=in_channels,
        )
        travel_final_auprc.append(hist_t[-1]["auprc"])
        fedavg_final_auprc.append(hist_f[-1]["auprc"])
        travel_histories.append(hist_t)
        fedavg_histories.append(hist_f)
    return {
        "travel_final_auprc": travel_final_auprc,
        "fedavg_final_auprc": fedavg_final_auprc,
        "travel_histories": travel_histories,
        "fedavg_histories": fedavg_histories,
    }
