from typing import Dict, Any, List, Callable, Union, Optional
import os

import numpy as np
import h5py

import torch

from .. import (
    AtomicData,
    AtomicDataDict,
)
from ..transforms import TypeMapper, OrbitalMapper
from ._base_datasets import AtomicDataset, AtomicInMemoryDataset
#from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data.interfaces.ham_to_feature import ham_block_to_feature
from dptb.utils.tools import j_loader

class _TrajData(object):

    def __init__(self, root: str, AtomicData_options: Dict[str, Any] = {},):
        self.root = root
        self.AtomicData_options = AtomicData_options
        self.info = j_loader(os.path.join(root, "info.json"))

        self.data = {}
        self.data["cell"] = np.loadtxt(os.path.join(root, "cell.dat"))
        self.data["atomic_numbers"] = np.loadtxt(os.path.join(root, "atomic_numbers.dat"))

        pos = np.loadtxt(os.path.join(root, "positions.dat"))
        assert pos.shape[0] == self.info["nframes"] * self.info["natoms"]
        self.data["pos"] = pos.reshape(self.info["nframes"], self.info["natoms"], 3)

        if os.path.exists(os.path.join(self.root, "eigenvalues.dat")):
            assert os.path.exists(os.path.join(self.root, "kpoints.dat"))
            self.data["kpoints"] = np.loadtxt(os.path.join(self.root, "kpoints.dat"))
            eigenvalues = np.loadtxt(os.path.join(self.root, "eigenvalues.dat"))
            assert eigenvalues.shape[0] == self.info["nframes"] * self.info["bandinfo"]["nkpoints"]
            assert eigenvalues.shape[1] == self.info["bandinfo"]["nbands"]
            self.data["eigenvalues"] = eigenvalues.reshape(self.info["nframes"], 
                                                           self.info["bandinfo"]["nkpoints"], 
                                                           self.info["bandinfo"]["nbands"])            
        if os.path.exists(os.path.join(self.root, "hamiltonians.h5")):
            self.data["hamiltonian_blocks"] = h5py.File(os.path.join(self.root, "hamiltonians.h5"), "r")
        if os.path.exists(os.path.join(self.root, "overlaps.h5")):
            self.data["overlap_blocks"] = h5py.File(os.path.join(self.root, "overlaps.h5"), "r")
        
    def toAtomicDataList(self):
        data_list = []
        for frame in range(self.info["nframes"]):
            atomic_data = AtomicData.from_points(
                pos = self.data["pos"][frame],
                cell = self.data["cell"],
                atomic_numbers = self.data["atomic_numbers"],
                pbc = self.info["pbc"],
                **self.AtomicData_options)
            if "hamiltonian_blocks" in self.data:
                if self.info.get("basis"):
                    idp = OrbitalMapper(self.info["basis"])
                else:
                    raise ValueError("Basis set must be provided for loading Hamiltonain.")
                if "overlap_blocks" not in self.data:
                    self.data["overlap_blocks"] = False
                # e3 = E3Hamiltonian(idp=idp, decompose=True)
                ham_block_to_feature(atomic_data, idp, 
                                     self.data["hamiltonian_blocks"][str(frame)], 
                                     self.data["overlap_blocks"][str(frame)])
                # with torch.no_grad():
                #     atomic_data = e3(atomic_data.to_dict())
                # atomic_data = AtomicData.from_dict(atomic_data)
            if "eigenvalues" in self.data and "kpoints" in self.data:
                bandinfo = self.info["bandinfo"]
                atomic_data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(self.data["kpoints"][:], 
                                                                         dtype=torch.get_default_dtype())
                if bandinfo["emin"] is not None and bandinfo["emax"] is not None:
                    atomic_data[AtomicDataDict.ENERGY_WINDOWS_KEY] = torch.as_tensor([bandinfo["emin"], bandinfo["emax"]], 
                                                                                     dtype=torch.get_default_dtype())
                if bandinfo["band_min"] is not None and bandinfo["band_max"] is not None:
                    atomic_data[AtomicDataDict.BAND_WINDOW_KEY] = torch.as_tensor([bandinfo["band_min"], bandinfo["band_max"]], 
                                                                                  dtype=torch.get_default_dtype())
                    atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(self.data["eigenvalues"][frame][bandinfo["band_min"]:bandinfo["band_max"]], 
                                                                                dtype=torch.get_default_dtype())
                else:
                    atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(self.data["eigenvalues"][frame], 
                                                                                dtype=torch.get_default_dtype())
            data_list.append(atomic_data)
        return data_list
        

class DefaultDataset(AtomicInMemoryDataset):

    def __init__(
            self,
            root: str,
            url: Optional[str] = None,
            AtomicData_options: Dict[str, Any] = {},
            include_frames: Optional[List[int]] = None,
            type_mapper: TypeMapper = None,
    ):
        self.file_name = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
        all_basis = []
        for file in self.file_name:
            file_info = j_loader(os.path.join(file, "info.json"))
            all_basis.append(file_info["basis"])
        sort_basis = {}
        for basis in all_basis:
            for symbol, orbitals in basis.items():
                if symbol not in sort_basis:
                    sort_basis[symbol] = orbitals
        type_mapper = OrbitalMapper(sort_basis)
        super().__init__(
            file_name=self.file_name,
            url=url,
            root=root,
            AtomicData_options=AtomicData_options,
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    def setup_data(self):
        self.data = []
        for file in self.file_name:
            subdata = _TrajData(os.path.join(self.root, file), self.AtomicData_options)
            self.data.append(subdata)

    def get_data(self):
        self.setup_data()
        all_data = []
        for subdata in self.data:
            subdata_list = subdata.toAtomicDataList()
            all_data += subdata_list
        return all_data
    
    @property
    def raw_file_names(self):
        return "Null"

    @property
    def raw_dir(self):
        return self.root
    