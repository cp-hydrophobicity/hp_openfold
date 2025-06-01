# # Import and expose key components from the MMC module
# from openfold.model.mmc.core import load_md_prediction_model
# from openfold.model.mmc.model import MMCRefinementModel, compute_energy_gradient, backprop_energy_gradient
# from openfold.model.mmc.data import create_data_loaders, create_unified_dataset
# from openfold.model.mmc.loss import RefinementLoss, rmsd_loss
# from openfold.model.mmc.metrics import calculate_ca_rmsd, calculate_atom14_rmsd

# # Avoid circular imports by lazily loading certain modules
# def get_evaluate_step():
#     from openfold.model.mmc.core import evaluate_step
#     return evaluate_step

# def get_run_module():
#     import openfold.model.mmc.run
#     return openfold.model.mmc.run

# __all__ = [
#     "load_md_prediction_model",
#     "get_evaluate_step",
#     "MMCRefinementModel",
#     "compute_energy_gradient",
#     "backprop_energy_gradient",
#     "create_data_loaders",
#     "create_unified_dataset",
#     "RefinementLoss",
#     "rmsd_loss",
#     "calculate_ca_rmsd",
#     "calculate_atom14_rmsd",
#     "get_run_module"
# ]
