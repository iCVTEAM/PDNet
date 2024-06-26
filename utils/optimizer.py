from model.pdnet import PDNet, PartBasisGenerator


def get_backbone_params(model: PDNet):
    for param in model.bkbone.parameters():
        if param.requires_grad:
            yield param


def get_head_params(model: PDNet, part_basis_generator: PartBasisGenerator):
    for name, param in model.named_parameters():
        if 'bkbone' not in name:
            yield param
    for param in part_basis_generator.parameters():
        yield param
