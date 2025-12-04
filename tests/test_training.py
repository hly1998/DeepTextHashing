#!/usr/bin/env python
"""
模型训练流程测试

用于验证所有模型的训练流程是否正常工作。
每个模型只训练少量epoch，使用小batch，快速验证代码正确性。

使用方法:
    cd /path/to/DeepTextHashing
    python -m tests.test_training
"""

import importlib.util
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from textdata import (
    SingleLabelTextDataset,
    MultiLabelTextDataset,
    SingleLabelTextDatasetDocID,
    MultiLabelTextDatasetDocID
)
from utils.utils import set_seed, retrieve_topk, compute_precision_at_k

# 测试配置
TEST_CONFIG = {
    'dataset': 'ng20',
    'data_fmt': 'tfidf',
    'batch_size': 32,
    'epochs': 2,
    'bit': 8,
    'lr': 0.001,
}


def get_device():
    """获取测试设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_test_data(with_doc_id=False):
    """获取测试数据"""
    data_path = PROJECT_ROOT / 'textdata' / TEST_CONFIG['dataset']

    if with_doc_id:
        train_set = SingleLabelTextDatasetDocID(
            str(data_path), subset='train', bow_format=TEST_CONFIG['data_fmt'])
        test_set = SingleLabelTextDatasetDocID(
            str(data_path), subset='test', bow_format=TEST_CONFIG['data_fmt'])
    else:
        train_set = SingleLabelTextDataset(
            str(data_path), subset='train', bow_format=TEST_CONFIG['data_fmt'])
        test_set = SingleLabelTextDataset(
            str(data_path), subset='test', bow_format=TEST_CONFIG['data_fmt'])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=TEST_CONFIG['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=TEST_CONFIG['batch_size'], shuffle=False)

    return train_set, test_set, train_loader, test_loader


def load_module_from_file(module_name: str, file_path: Path):
    """从文件路径动态加载模块（支持带连字符的目录名）"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestResult:
    """测试结果"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.success = False
        self.error = None
        self.final_loss = None
        self.final_prec = None

    def __str__(self):
        if self.success:
            return f"✓ {self.model_name}: loss={self.final_loss:.4f}, prec={self.final_prec:.4f}"
        else:
            return f"✗ {self.model_name}: {self.error}"


def test_vdsh():
    """测试 VDSH 模型"""
    from models.VDSH.VDSH import VDSH

    result = TestResult("VDSH")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)

        model = VDSH(TEST_CONFIG['dataset'], num_features, TEST_CONFIG['bit'], device, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                logprob_w, mu, logvar = model(xb)
                kl_loss = VDSH.calculate_KL_loss(mu, logvar)
                reconstr_loss = VDSH.compute_reconstr_loss(logprob_w, xb)
                loss = reconstr_loss + 0.1 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        # 评估
        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_vdsh_s():
    """测试 VDSH_S 模型"""
    from models.VDSH.VDSH_S import VDSH_S

    result = TestResult("VDSH_S")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)
        y_dim = train_set.num_classes()

        model = VDSH_S(TEST_CONFIG['dataset'], num_features, TEST_CONFIG['bit'], y_dim, device, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logprob_w, mu, logvar, score_c = model(xb)
                kl_loss = VDSH_S.calculate_KL_loss(mu, logvar)
                reconstr_loss = VDSH_S.compute_reconstr_loss(logprob_w, xb)
                pred_loss = model.compute_prediction_loss(score_c, yb)
                loss = reconstr_loss + 0.1 * kl_loss + 0.1 * pred_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_wish():
    """测试 WISH 模型"""
    from models.WISH.WISH import WISH

    result = TestResult("WISH")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)
        num_bits = TEST_CONFIG['bit']

        model = WISH(TEST_CONFIG['dataset'], num_features, num_bits, 100, num_bits, device, dropoutProb=0.2)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])
        I_matrix = torch.eye(num_bits).to(device)

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                logprob_w, mu, topicS = model(xb, True, integration='sum')
                kl_loss = model.calculate_KL_loss(mu)
                reconstr_loss = WISH.compute_reconstr_loss(logprob_w, xb)
                loss = reconstr_loss + 0.1 * kl_loss
                loss += torch.pow(torch.norm(topicS - I_matrix), 2) * 1.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader, isStochastic=True)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_wish_s():
    """测试 WISH_S 模型"""
    from models.WISH.WISH_S import WISH_S

    result = TestResult("WISH_S")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)
        num_bits = TEST_CONFIG['bit']
        y_dim = train_set.num_classes()

        model = WISH_S(TEST_CONFIG['dataset'], num_features, num_bits, 100, num_bits, y_dim, device, dropoutProb=0.2)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])
        I_matrix = torch.eye(num_bits).to(device)

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logprob_w, mu, topicS, score_c = model(xb, True, integration='sum')
                kl_loss = model.calculate_KL_loss(mu)
                reconstr_loss = WISH_S.compute_reconstr_loss(logprob_w, xb)
                pred_loss = model.compute_prediction_loss(score_c, yb)
                loss = reconstr_loss + 0.1 * kl_loss + 0.1 * pred_loss
                loss += torch.pow(torch.norm(topicS - I_matrix), 2) * 1.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader, isStochastic=True)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_nash():
    """测试 NASH 模型"""
    from models.NASH.NASH import NASH

    result = TestResult("NASH")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)

        model = NASH(num_features, TEST_CONFIG['bit'], device, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                logprob_w, mu = model(xb, isStochastic=True)
                kl_loss = model.calculate_KL_loss(mu)
                reconstr_loss = NASH.compute_reconstr_loss(logprob_w, xb)
                loss = reconstr_loss + 0.1 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_nash_s():
    """测试 NASH_S 模型"""
    from models.NASH.NASH_S import NASH_S

    result = TestResult("NASH_S")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)
        y_dim = train_set.num_classes()

        model = NASH_S(num_features, TEST_CONFIG['bit'], y_dim, device, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logprob_w, mu, score_c = model(xb, isStochastic=True)
                kl_loss = model.calculate_KL_loss(mu)
                reconstr_loss = NASH_S.compute_reconstr_loss(logprob_w, xb)
                pred_loss = model.compute_prediction_loss(score_c, yb)
                loss = reconstr_loss + 0.1 * kl_loss + 0.1 * pred_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_doc2hash():
    """测试 Doc2Hash 模型"""
    from models.Doc2Hash.Doc2Hash import Doc2Hash

    result = TestResult("Doc2Hash")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)

        model = Doc2Hash(TEST_CONFIG['dataset'], num_features, TEST_CONFIG['bit'], device, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])
        tmp = 1.0

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                logprob_w, qy = model(xb, tmp)
                kl_loss = Doc2Hash.calculate_KL_loss(qy)
                reconstr_loss = Doc2Hash.compute_reconstr_loss(logprob_w, xb)
                loss = reconstr_loss + 2.0 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tmp = max(tmp * 0.96, 0.1)
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_doc2hash_s():
    """测试 Doc2Hash_S 模型"""
    from models.Doc2Hash.Doc2Hash_S import Doc2Hash_S

    result = TestResult("Doc2Hash_S")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)
        y_dim = train_set.num_classes()

        model = Doc2Hash_S(TEST_CONFIG['dataset'], num_features, TEST_CONFIG['bit'], y_dim, device, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])
        tmp = 1.0

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logprob_w, qy, score_c = model(xb, tmp)
                kl_loss = Doc2Hash_S.calculate_KL_loss(qy)
                reconstr_loss = Doc2Hash_S.compute_reconstr_loss(logprob_w, xb)
                pred_loss = model.compute_prediction_loss(score_c, yb)
                loss = reconstr_loss + 2.0 * kl_loss + 0.1 * pred_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tmp = max(tmp * 0.96, 0.1)
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_smash():
    """测试 SMASH 模型"""
    from models.SMASH.SMASH import SMASH

    result = TestResult("SMASH")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data(with_doc_id=True)
        num_features = train_set[0][0].size(0)
        num_bits = TEST_CONFIG['bit']
        time_max = int(len(train_set) / TEST_CONFIG['batch_size'])

        model = SMASH(TEST_CONFIG['dataset'], num_features, num_bits, len(train_set), time_max,
                      em_alpha=0.3, device=device, sigma=0.3, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, idxs, yb in train_loader:
                xb = xb.to(device)
                logprob_w, _, z, _, long_z, _, em_out = model(xb, idxs, epoch, 0.5, n_sample=3)
                rec_loss = -torch.mean(torch.sum(logprob_w * xb, dim=1))
                loss = rec_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_smash_s():
    """测试 SMASH_S 模型"""
    from models.SMASH.SMASH_S import SMASH_S

    result = TestResult("SMASH_S")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data(with_doc_id=True)
        num_features = train_set[0][0].size(0)
        num_bits = TEST_CONFIG['bit']
        time_max = int(len(train_set) / TEST_CONFIG['batch_size'])
        y_dim = train_set.num_classes()

        model = SMASH_S(TEST_CONFIG['dataset'], num_features, num_bits, len(train_set), time_max,
                        em_alpha=0.3, num_classes=y_dim, device=device, sigma=0.3, dropoutProb=0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, idxs, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logprob_w, _, z, _, long_z, _, em_out, score_c = model(xb, idxs, epoch, 0.5, n_sample=3)
                rec_loss = -torch.mean(torch.sum(logprob_w * xb, dim=1))
                pred_loss = model.compute_prediction_loss(score_c, yb)
                loss = rec_loss + 0.1 * pred_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


def test_bvae():
    """测试 B-VAE 模型"""
    # 由于目录名包含连字符，需要动态加载
    bvae_path = PROJECT_ROOT / 'models' / 'B-VAE' / 'BVAE.py'
    bvae_module = load_module_from_file('BVAE', bvae_path)
    BVAE = bvae_module.BVAE

    result = TestResult("B-VAE")
    device = get_device()

    try:
        train_set, test_set, train_loader, test_loader = get_test_data()
        num_features = train_set[0][0].size(0)

        model = BVAE(num_features, TEST_CONFIG['bit'], device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG['lr'])

        for epoch in range(TEST_CONFIG['epochs']):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                output, logits_b = model(xb)
                kl_loss = BVAE.bkl_loss(logits_b)
                reconstr_loss = BVAE.rec_loss(xb, output)
                loss = torch.mean(reconstr_loss + 0.0625 * kl_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            result.final_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=True)
            result.final_prec = prec.item()

        result.success = True
    except Exception as e:
        result.error = str(e)

    return result


# 所有测试函数
ALL_TESTS = [
    test_vdsh,
    test_vdsh_s,
    test_wish,
    test_wish_s,
    test_nash,
    test_nash_s,
    test_doc2hash,
    test_doc2hash_s,
    test_smash,
    test_smash_s,
    test_bvae,
]


def run_all_tests():
    """运行所有测试"""
    set_seed()
    device = get_device()

    print("=" * 60)
    print("Deep Text Hashing 模型训练测试")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"数据集: {TEST_CONFIG['dataset']}.{TEST_CONFIG['data_fmt']}")
    print(f"测试 epochs: {TEST_CONFIG['epochs']}")
    print(f"Batch size: {TEST_CONFIG['batch_size']}")
    print("=" * 60)

    results = []
    for test_func in ALL_TESTS:
        print(f"\n正在测试 {test_func.__doc__.strip()}...")
        result = test_func()
        results.append(result)
        print(result)

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    for result in results:
        print(result)

    print("-" * 60)
    print(f"总计: {passed} 通过, {failed} 失败, 共 {len(results)} 个模型")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
