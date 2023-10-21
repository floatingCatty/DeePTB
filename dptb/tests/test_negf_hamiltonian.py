# Hamiltonian
from dptb.plugins.init_nnsk import InitSKModel
from dptb.nnops.NN2HRK import NN2HRK
from dptb.nnops.apihost import NNSKHost
from dptb.utils.tools import j_must_have
from dptb.utils.tools import j_loader
import numpy as np
import torch
import pytest
from dptb.negf.hamiltonian import Hamiltonian
from ase.io import read


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


def test_negf_Hamiltonian(root_directory):

    model_ckpt=root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
    jdata = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_config.json"
    structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
    log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/test.log"

    apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
    apihost.register_plugin(InitSKModel())
    apihost.build()
    apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
    jdata = j_loader(jdata)
    task_options = j_must_have(jdata, "task_options")

    run_opt = {
            "run_sk": True,
            "init_model":model_ckpt,
            "results_path":root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/",
            "structure":structure,
            "log_path": log_path,
            "log_level": 5,
            "use_correction":False
        }


    structase=read(run_opt['structure'])
    results_path=run_opt.get('results_path')
    kpoints=np.array([[0,0,0]])

    hamiltonian = Hamiltonian(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    with torch.no_grad():
        struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)
    
    #check device's Hamiltonian
    device_symbol= struct_device.symbols=="C4"
    device_pbc= struct_device.pbc==False
    device_cell= struct_device.cell==[10.0, 10.0, 19.2]
    assert device_symbol.any()==True
    assert device_pbc.any()==True
    assert device_cell.any()==True

    #check lead_L's Hamiltonian
    leads_symbol= struct_leads["lead_L"].symbols=="C4"
    leads_pbc= struct_leads["lead_L"].pbc==False
    leads_cell= struct_leads["lead_L"].cell==[10.0, 10.0, 6.4]
    assert leads_symbol.any()==True
    assert leads_pbc.any()==True
    assert leads_cell.any()==True  
    
    #check lead_R's Hamiltonian
    leads_symbol= struct_leads["lead_R"].symbols=="C4"
    leads_pbc= struct_leads["lead_R"].pbc==False
    leads_cell= struct_leads["lead_R"].cell==[10.0, 10.0, 6.4]
    assert leads_symbol.any()==True
    assert leads_pbc.any()==True
    assert leads_cell.any()==True 

    #check hs_device
    hs_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0][0]
    hs_device_standard = torch.tensor([[-13.6386+0.j,   0.6096+0.j,   0.0000+0.j,   0.0000+0.j],
        [  0.6096+0.j, -13.6386+0.j,   0.6096+0.j,   0.0000+0.j],
        [  0.0000+0.j,   0.6096+0.j, -13.6386+0.j,   0.6096+0.j],
        [  0.0000+0.j,   0.0000+0.j,   0.6096+0.j, -13.6386+0.j]],dtype=torch.complex128)
    assert abs(hs_device-hs_device_standard).all()<1e-4

    #check hs_lead
    hs_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[0][0]
    hs_lead_standard = torch.tensor([-13.6386+0.j,   0.6096+0.j], dtype=torch.complex128)    
    assert abs(hs_lead-hs_lead_standard).max()<1e-4

   
