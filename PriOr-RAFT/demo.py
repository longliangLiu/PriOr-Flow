import argparse
import torch
from core.prior_raft import PriOr_RAFT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()

    model = PriOr_RAFT(args)
    model.cuda()
    model.eval()

    image1 = torch.randn(1, 3, 512, 1024).cuda()
    image2 = torch.randn(1, 3, 512, 1024).cuda()

    with torch.no_grad():
        flow_pr = model(image1, image2, iters=12, test_mode=True)

    print(flow_pr.shape)