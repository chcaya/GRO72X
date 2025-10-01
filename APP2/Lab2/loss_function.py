import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self, alpha):
        super(LocalizationLoss, self).__init__()
        self._alpha = alpha

    def forward(self, output, target):
        # ------------------------ Laboratoire 2 - Question 4 - Début de la section à compléter ------------------------
        # À compléter
        return torch.tensor(0.0)
        # ------------------------ Laboratoire 2 - Question 4 - Fin de la section à compléter --------------------------



def check_loss_output_size(loss):
    assert loss.size() == torch.Size([]), f'La sortie de la fonction de coût ({loss.item()}) doit être un scalaire.'


def check_loss_output_near(loss, expect_loss):
    assert abs(loss.item() - expect_loss) < 1e-3, f'La sortie de la fonction de coût ({loss.item()}) n\'est pas proche de {expect_loss}.'


def check_loss_output_smaller_than(loss1, loss2):
    assert loss1.item() < loss2.item(), f'La sortie de la fonction de coût ({loss1.item()}) n\'est pas plus petite que de {loss2.item()}.'


def test_reduction():
    print("Test - Reduction")
    criterion = LocalizationLoss(alpha=2)
    output = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                           [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
    target = torch.tensor([[0.4, 0.35, 0.6, 0.65, 1.0],
                           [0.4, 0.35, 0.6, 0.65, 0.0]])

    loss = criterion(output, target)
    check_loss_output_size(loss)
    print("\tOk")


def test_perfect_output():
    print("Test - Perfect output")
    criterion = LocalizationLoss(alpha=2)

    output = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    target = torch.tensor([[0.4, 0.35, 0.6, 0.65, 2.0]])

    loss = criterion(output, target)
    check_loss_output_near(loss, 0.0)
    print("\tOk")


def test_alpha_class_ok():
    print("Test - Alpha when the class is ok")
    criterion_big_alpha = LocalizationLoss(alpha=2)
    criterion_small_alpha = LocalizationLoss(alpha=1)

    output = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10]])
    target = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.0]])

    loss_big_alpha = criterion_big_alpha(output, target)
    loss_small_alpha = criterion_small_alpha(output, target)
    check_loss_output_near(loss_big_alpha, loss_small_alpha.item())
    print("\tOk")


def test_alpha_class_not_ok():
    print("Test - Alpha when the class is not ok")
    criterion_big_alpha = LocalizationLoss(alpha=2)
    criterion_small_alpha = LocalizationLoss(alpha=1)

    output = torch.tensor([[0.1, 0.3, 0.2, 0.4, 0.0, 1e3, 0.0]])
    target = torch.tensor([[0.0, 0.1, 0.2, 0.5, 0.0]])

    loss_big_alpha = criterion_big_alpha(output, target)
    loss_small_alpha = criterion_small_alpha(output, target)
    check_loss_output_near(loss_big_alpha, 2 * loss_small_alpha.item())
    print("\tOk")


def test_box():
    print("Test - Box")
    criterion = LocalizationLoss(alpha=2)

    output_perfect = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_x_not_perfect = torch.tensor([[0.52, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_y_not_perfect = torch.tensor([[0.5, 0.4, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_w_not_perfect = torch.tensor([[0.5, 0.5, 0.1, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_h_not_perfect = torch.tensor([[0.5, 0.5, 0.2, 0.5, 0.0, 0.0, 1e10, 0.0]])
    target = torch.tensor([[0.4, 0.35, 0.6, 0.65, 2.0]])

    loss_perfect = criterion(output_perfect, target)
    check_loss_output_smaller_than(loss_perfect, criterion(output_x_not_perfect, target))
    check_loss_output_smaller_than(loss_perfect, criterion(output_y_not_perfect, target))
    check_loss_output_smaller_than(loss_perfect, criterion(output_w_not_perfect, target))
    check_loss_output_smaller_than(loss_perfect, criterion(output_h_not_perfect, target))
    print("\tOk")


if __name__ == '__main__':
    test_reduction()
    test_perfect_output()
    test_alpha_class_ok()
    test_alpha_class_not_ok()
    test_box()
