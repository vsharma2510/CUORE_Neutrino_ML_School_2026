import uproot
import pandas as pd
import awkward as ak
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

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

                for datasetName in ("energy", "npulse"):
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
