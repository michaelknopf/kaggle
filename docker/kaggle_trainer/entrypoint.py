from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.sagemaker_ops import train_entrypoint

train_entrypoint.main(sm_utils.model_name)
