import uproot
import pandas as pd
import awkward as ak
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None

    class TorchDataset:
        """Fallback base class so the module can be imported without PyTorch."""

        pass

def printRootBranches(rootFile, treeName):
    """
    Open a ROOT file and print all branch names with their data types.

    Parameters
    ----------
    rootFile : str
        Path to the ROOT file
    treeName : str
        Name of the TTree
    """

    try:
        with uproot.open(rootFile) as rootHandle:
            tree = rootHandle[treeName]

            print(f"\nBranches in tree '{treeName}':\n")

            for branchName, branch in tree.items():
                print(f"{branchName:30s} {branch.typename}")

    except KeyError:
        print(f"Tree '{treeName}' not found in {rootFile}")

    except Exception as e:
        print(f"Error opening ROOT file: {e}")

def rootBranchesToHdf5(
    rootFile,
    treeName,
    branchList,
    hdf5File,
    entryStart=None,
    entryStop=None
):
    """
    Read selected branches from a ROOT TTree and store them in an HDF5 file.

    Scalar branches -> stored as 1D datasets
    std::vector branches -> stored as 2D datasets (nEntries, vectorLength)

    Parameters
    ----------
    rootFile : str
        Path to the input ROOT file
    treeName : str
        Name of the TTree inside the ROOT file
    branchList : list[str]
        Branch names to read
    hdf5File : str
        Path to the output HDF5 file
    entryStart : int or None
        First entry to read
    entryStop : int or None
        Last entry to read
    """

    try:
        with uproot.open(rootFile) as rootHandle:
            tree = rootHandle[treeName]

            with h5py.File(hdf5File, "w") as hdf5Handle:

                for branchName in branchList:

                    branch = tree[branchName]
                    branchType = branch.typename

                    # ---- VECTOR BRANCH ----
                    if "vector<" in branchType:

                        data = branch.array(
                            library="ak",
                            entry_start=entryStart,
                            entry_stop=entryStop
                        )

                        data = ak.to_regular(data)
                        data = ak.to_numpy(data)

                        hdf5Handle.create_dataset(
                            branchName,
                            data=data,
                            compression="gzip"
                        )

                    # ---- SCALAR BRANCH ----
                    else:

                        data = branch.array(
                            library="np",
                            entry_start=entryStart,
                            entry_stop=entryStop
                        )

                        data = np.asarray(data)

                        hdf5Handle.create_dataset(
                            branchName,
                            data=data,
                            compression="gzip"
                        )

    except KeyError:
        print(f"Tree '{treeName}' not found in {rootFile}")

    except Exception as e:
        print(f"Error converting ROOT file '{rootFile}': {e}")

def inspectHdf5Entry(hdf5File, entryIndex, plotBranch=None, xValues=None, maxCols=3):
    """
    Read one or more entries from an HDF5 file, print the values for those entries,
    and optionally plot one array-valued branch.

    Parameters
    ----------
    hdf5File : str
        Path to the HDF5 file.
    entryIndex : int or sequence of int
        Entry index or collection of entry indices to inspect.
    plotBranch : str or None
        Name of the branch to plot. This should usually be a vector-valued dataset.
        If None, no plot is made.
    xValues : array-like or None
        Optional x-axis values for the plot. Must have the same length as each
        selected array entry.
    maxCols : int
        Maximum number of subplot columns when plotting multiple entries.
    """

    try:
        with h5py.File(hdf5File, "r") as hdf5Handle:
            datasetNames = list(hdf5Handle.keys())

            def formatMetadataValue(value):
                if np.isscalar(value) or np.ndim(value) == 0:
                    scalarValue = np.asarray(value).item()
                    if isinstance(scalarValue, (float, np.floating)):
                        return f"{scalarValue:g}"
                    return str(scalarValue)
                return None

            def buildEntryTitle(index, prefix):
                metadataParts = []

                for datasetName in ("Energy", "NPulses"):
                    if datasetName not in hdf5Handle:
                        continue

                    formattedValue = formatMetadataValue(hdf5Handle[datasetName][index])
                    if formattedValue is not None:
                        metadataParts.append(f"{datasetName}={formattedValue}")

                if metadataParts:
                    return f"{prefix} ({', '.join(metadataParts)})"

                return prefix

            if len(datasetNames) == 0:
                raise ValueError(f"No datasets found in {hdf5File}")

            firstDataset = hdf5Handle[datasetNames[0]]
            nEntries = firstDataset.shape[0]

            if np.isscalar(entryIndex):
                entryList = [int(entryIndex)]
            else:
                entryList = [int(index) for index in entryIndex]

            if len(entryList) == 0:
                raise ValueError("entryIndex list is empty")

            for index in entryList:
                if index < 0 or index >= nEntries:
                    raise IndexError(
                        f"entryIndex {index} is out of range. "
                        f"Valid range is 0 to {nEntries - 1}"
                    )

            print(f"Inspecting entries {entryList} in {hdf5File}\n")

            # for index in entryList:
            #     print(f"===== Entry {index} =====")

            #     for datasetName in datasetNames:
            #         data = hdf5Handle[datasetName]
            #         entryValue = data[index]

            #         print(f"{datasetName}:")
            #         # print(f"  shape of dataset = {data.shape}")
            #         # print(f"  dtype = {data.dtype}")

            #         if np.isscalar(entryValue) or np.ndim(entryValue) == 0:
            #             print(f"  value = {entryValue}\n")
            #         else:
            #             entryArray = np.asarray(entryValue)
            #             print(f"  entry shape = {entryArray.shape}")
            #             # print(f"  first few values = {entryArray[:10]}\n")

            if plotBranch is not None:
                if plotBranch not in hdf5Handle:
                    raise KeyError(f"Branch '{plotBranch}' not found in {hdf5File}")

                dataset = hdf5Handle[plotBranch]

                if xValues is not None:
                    xValues = np.asarray(xValues)

                if len(entryList) == 1:
                    plotData = np.asarray(dataset[entryList[0]])

                    if plotData.ndim == 0:
                        raise ValueError(
                            f"Branch '{plotBranch}' at entry {entryList[0]} is a scalar, not an array"
                        )

                    if xValues is not None and len(xValues) != len(plotData):
                        raise ValueError(
                            f"xValues has length {len(xValues)}, but plot data has length {len(plotData)}"
                        )

                    plt.figure(figsize=(10, 5))

                    if xValues is not None:
                        plt.plot(xValues, plotData)
                    else:
                        plt.plot(plotData)

                    plt.xlabel("Index" if xValues is None else "x")
                    plt.ylabel(plotBranch)
                    plt.title(buildEntryTitle(entryList[0], f"{plotBranch} for entry {entryList[0]}"))
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                else:
                    nPlots = len(entryList)
                    nCols = min(maxCols, nPlots)
                    nRows = math.ceil(nPlots / nCols)

                    fig, axes = plt.subplots(
                        nRows,
                        nCols,
                        figsize=(5 * nCols, 3.5 * nRows),
                        squeeze=False
                    )

                    flatAxes = axes.flatten()

                    for axis, index in zip(flatAxes, entryList):
                        plotData = np.asarray(dataset[index])

                        if plotData.ndim == 0:
                            raise ValueError(
                                f"Branch '{plotBranch}' at entry {index} is a scalar, not an array"
                            )

                        if xValues is not None and len(xValues) != len(plotData):
                            raise ValueError(
                                f"xValues has length {len(xValues)}, but plot data has length {len(plotData)}"
                            )

                        if xValues is not None:
                            axis.plot(xValues, plotData)
                        else:
                            axis.plot(plotData)

                        axis.set_title(buildEntryTitle(index, f"Entry {index}"))
                        axis.set_xlabel("Index" if xValues is None else "x")
                        axis.set_ylabel(plotBranch)
                        axis.grid(True)

                    for axis in flatAxes[nPlots:]:
                        axis.axis("off")

                    #fig.suptitle(f"{plotBranch} for selected entries", fontsize=14)
                    plt.tight_layout()
                    plt.show()

    except Exception as e:
        print(f"Error reading HDF5 file '{hdf5File}': {e}")


def subtractWaveformMean(sample, waveformKey="Waveform"):
    """
    Subtract the mean of a waveform from the waveform values in one sample.

    Parameters
    ----------
    sample : dict
        One sample produced by `Hdf5TorchDataset`.
    waveformKey : str
        Key in the sample dictionary containing the waveform.
    """

    if waveformKey not in sample:
        raise KeyError(f"Waveform key '{waveformKey}' not found in sample")

    waveform = sample[waveformKey]

    if torch is not None and isinstance(waveform, torch.Tensor):
        sample[waveformKey] = waveform - waveform.mean()
        return sample

    waveformArray = np.asarray(waveform)
    sample[waveformKey] = waveformArray - waveformArray.mean()
    return sample


class Hdf5TorchDataset(TorchDataset):
    """
    PyTorch-style Dataset backed by an HDF5 file.

    Parameters
    ----------
    hdf5File : str or Path
        Path to the HDF5 file.
    fields : sequence of str or None
        Dataset names to read from the file. If None, all datasets are returned
        using their names exactly as they appear in the HDF5 file.
    tensorFields : sequence of str or None
        Dataset names to convert to torch tensors. If None, all numeric fields are
        converted when `convertToTensor=True`.
    convertToTensor : bool
        If True, convert numeric arrays/scalars to torch tensors.
    includeEntryIndex : bool
        If True, include the selected entry index in each returned sample.
    sampleTransform : callable or None
        Optional transform applied to the assembled sample dictionary.
    """

    def __init__(
        self,
        hdf5File,
        fields=None,
        tensorFields=None,
        convertToTensor=True,
        includeEntryIndex=False,
        sampleTransform=subtractWaveformMean,
    ):
        self.hdf5File = Path(hdf5File)
        self.convertToTensor = convertToTensor
        self.includeEntryIndex = includeEntryIndex
        self.sampleTransform = sampleTransform

        if self.convertToTensor and torch is None:
            raise ImportError(
                "PyTorch is required when convertToTensor=True, but torch is not installed."
            )

        with h5py.File(self.hdf5File, "r") as hdf5Handle:
            self.availableKeys = self._listDatasetPaths(hdf5Handle)
            self._entryCount = self._inferEntryCount(hdf5Handle)
            self.fields = self._resolveFields(fields)

        self.tensorFields = None if tensorFields is None else set(tensorFields)
        self._hdf5Handle = None

    def __len__(self):
        return self._entryCount

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self)}")

        hdf5Handle = self._getHandle()
        sample = {}

        for fieldName in self.fields:
            value = np.asarray(hdf5Handle[fieldName][index])
            sample[fieldName] = self._convertValue(fieldName, value)

        if self.includeEntryIndex:
            sample["entryIndex"] = index

        if self.sampleTransform is not None:
            sample = self.sampleTransform(sample)

        return sample

    def close(self):
        if self._hdf5Handle is not None:
            self._hdf5Handle.close()
            self._hdf5Handle = None

    def resolvedKeys(self):
        """Return the dataset names read from the HDF5 file."""
        return list(self.fields)

    def _getHandle(self):
        if self._hdf5Handle is None:
            self._hdf5Handle = h5py.File(self.hdf5File, "r")
        return self._hdf5Handle

    def _convertValue(self, fieldName, value):
        if value.ndim == 0:
            value = value.item()

        if not self.convertToTensor:
            return value

        if self.tensorFields is not None and fieldName not in self.tensorFields:
            return value

        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            return torch.from_numpy(value)

        if isinstance(value, (np.generic, int, float, bool)):
            return torch.as_tensor(value)

        return value

    def _resolveFields(self, fields):
        if fields is None:
            return self.availableKeys

        resolvedFields = [str(fieldName) for fieldName in fields]
        missingFields = [
            fieldName for fieldName in resolvedFields if fieldName not in self.availableKeys
        ]

        if missingFields:
            raise KeyError(
                f"Fields {missingFields} were not found in {self.hdf5File}. "
                f"Available datasets: {self.availableKeys}"
            )

        return resolvedFields

    @staticmethod
    def _listDatasetPaths(hdf5Handle):
        datasetPaths = []
        hdf5Handle.visititems(
            lambda name, obj: datasetPaths.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        return sorted(datasetPaths)

    @staticmethod
    def _inferEntryCount(hdf5Handle):
        datasetPaths = Hdf5TorchDataset._listDatasetPaths(hdf5Handle)
        if len(datasetPaths) == 0:
            raise ValueError("No datasets found in HDF5 file")

        entryCount = None

        for datasetPath in datasetPaths:
            dataset = hdf5Handle[datasetPath]

            if len(dataset.shape) == 0:
                continue

            currentCount = dataset.shape[0]

            if entryCount is None:
                entryCount = currentCount
            elif currentCount != entryCount:
                raise ValueError(
                    "Datasets in the HDF5 file do not agree on the number of entries. "
                    f"Found {entryCount} and {currentCount}."
                )

        if entryCount is None:
            raise ValueError("Could not infer dataset length from scalar-only HDF5 file")

        return entryCount

    def __del__(self):
        self.close()
