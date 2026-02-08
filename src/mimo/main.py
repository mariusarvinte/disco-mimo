import torch


def main():
    device = "cuda:0"
    num_rx, num_tx, num_pilots = 64, 32, 16

    H = torch.randn(num_rx, num_tx, dtype=torch.complex64, device=device)
    P = torch.randn(num_tx, num_pilots, dtype=torch.complex64, device=device)
    N = torch.randn(num_rx, num_pilots, dtype=torch.complex64, device=device)

    Y = torch.matmul(H, P) + N
    return Y


if __name__ == "__main__":
    main()
