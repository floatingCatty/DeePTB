#test_negf_Device_set_leadLR
from dptb.negf.Device import Device
from dptb.plugins.init_nnsk import InitSKModel
from dptb.nnops.NN2HRK import NN2HRK
from dptb.nnops.apihost import NNSKHost
from dptb.utils.tools import j_must_have
from dptb.utils.tools import j_loader
import numpy as np
import torch
from dptb.negf.hamiltonian import Hamiltonian
from ase.io import read
from dptb.negf.Lead import Lead
from dptb.utils.constants import *
import pytest


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_negf_Device(root_directory):
    model_ckpt=root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
    jdata = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_config.json"
    structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
    log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_Device/test.log"


    # read input files and generate Hamiltonian
    apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
    apihost.register_plugin(InitSKModel())
    apihost.build()
    apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
    jdata = j_loader(jdata)
    task_options = j_must_have(jdata, "task_options")

    run_opt = {
            "run_sk": True,
            "init_model":model_ckpt,
            "results_path":root_directory +"/dptb/tests/data/test_negf/test_negf_Device/",
            "structure":structure,
            "log_path": log_path,
            "log_level": 5,
            "use_correction":False
        }


    structase=read(run_opt['structure'])
    results_path=run_opt.get('results_path')
    kpoints=np.array([[0,0,0]])
    ele_T = task_options["ele_T"]
    kBT = k * ele_T / eV
    e_fermi = task_options["e_fermi"]

    hamiltonian = Hamiltonian(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    with torch.no_grad():
        struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)


    device = Device(hamiltonian, struct_device, results_path=results_path, efermi=e_fermi)
    device.set_leadLR(
                    lead_L=Lead(
                    hamiltonian=hamiltonian, 
                    tab="lead_L", 
                    structure=struct_leads["lead_L"], 
                    results_path=results_path,
                    e_T=ele_T,
                    efermi=e_fermi, 
                    voltage=task_options["stru_options"]["lead_L"]["voltage"]
                ),
                    lead_R=Lead(
                        hamiltonian=hamiltonian, 
                        tab="lead_R", 
                        structure=struct_leads["lead_R"], 
                        results_path=results_path, 
                        e_T=ele_T,
                        efermi=e_fermi, 
                        voltage=task_options["stru_options"]["lead_R"]["voltage"]
                )
            )
    
    # check device.Lead_L.structure
    assert np.array([device.lead_L.structure.symbols=='C4']).all()
    assert np.array([device.lead_L.structure.pbc==[False, False, True]]).all()
    assert np.array([(device.lead_L.structure.cell-np.diag([10.0, 10.0, 6.4])<1e-5)]).all()
    assert device.lead_L.tab=="lead_L"
    assert abs(device.mu+13.638587951660156)<1e-5
    # check device.Lead_R.structure
    assert np.array([device.lead_R.structure.symbols=='C4']).all()
    assert np.array([device.lead_R.structure.pbc==[False, False, True]]).all()
    assert np.array([(device.lead_R.structure.cell-np.diag([10.0, 10.0, 6.4])<1e-5)]).all()
    assert device.lead_R.tab=="lead_R"
    assert abs(device.mu+13.638587951660156)<1e-5


    # calculate Self energy and Green function
    stru_options = j_must_have(task_options, "stru_options")
    leads = stru_options.keys()
    for ll in leads:
        if ll.startswith("lead"): #calculate surface green function at E=0
            getattr(device, ll).self_energy(
                e=torch.tensor([0]), 
                kpoint=kpoints[0], 
                eta_lead=task_options["eta_lead"],
                method=task_options["sgf_solver"]
                )

    device.green_function(  e=torch.tensor([0]),   #calculate device green function at E=0
                            kpoint=kpoints[0], 
                            eta_device=task_options["eta_device"], 
                            block_tridiagonal=task_options["block_tridiagonal"]
                            )

    #check calculated green functions' results
    assert list(device.green.keys())==['g_trans', 'grd', 'grl', 'gru', 'gr_left', 'gnd', 'gnl',\
                                        'gnu', 'gin_left', 'gpd', 'gpl', 'gpu', 'gip_left']
    g_trans= torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
            [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
            [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
            [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)
    grd= [torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
            [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
            [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
            [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)]

    assert  np.array(abs(g_trans-device.green['g_trans'])<1e-5).all()
    assert  np.array(abs(grd[0]-device.green['grd'][0])<1e-5).all()
    assert device.green['grl'] == []
    assert device.green['gru'] == []

    gr_left= [torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
            [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
            [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
            [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)]

    gnd = [torch.tensor([[-2.2316e-07-4.1011e-01j,  1.2157e-13+2.2317e-07j,2.2316e-07+4.1011e-01j, -3.6432e-13-6.6949e-07j],
            [ 1.2132e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j, 1.2154e-13+2.2315e-07j,  2.2316e-07+4.1011e-01j],
            [ 2.2316e-07+4.1011e-01j,  1.2132e-13+2.2315e-07j,-2.2316e-07-4.1011e-01j,  1.2149e-13+2.2317e-07j],
            [-3.6429e-13-6.6949e-07j,  2.2316e-07+4.1011e-01j,1.2140e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j]],dtype=torch.complex128)]

    assert  np.array(abs(gr_left[0]-device.green['gr_left'][0])<1e-5).all()
    assert  np.array(abs(gnd[0]-device.green['gnd'][0])<1e-5).all()
    assert device.green['gnl'] == []
    assert device.green['gnu'] == []

    gin_left=[torch.tensor([[-2.2316e-07-4.1011e-01j,  1.2157e-13+2.2317e-07j,2.2316e-07+4.1011e-01j, -3.6432e-13-6.6949e-07j],
            [ 1.2132e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j,1.2154e-13+2.2315e-07j,  2.2316e-07+4.1011e-01j],
            [ 2.2316e-07+4.1011e-01j,  1.2132e-13+2.2315e-07j,-2.2316e-07-4.1011e-01j,  1.2149e-13+2.2317e-07j],
            [-3.6429e-13-6.6949e-07j,  2.2316e-07+4.1011e-01j,1.2140e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j]],dtype=torch.complex128)]
    assert  np.array(abs(gin_left[0]-device.green['gin_left'][0])<1e-5).all()

    assert device.green['gpd']== None
    assert device.green['gpl']== None
    assert device.green['gpu']== None
    assert device.green['gip_left']== None

    Tc=device._cal_tc_()
    assert abs(Tc-1)<1e-5




