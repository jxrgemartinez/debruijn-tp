#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib.pyplot as plt
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
from typing import Iterator, Dict, List


__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    Args:
        fastq_file (Path): Path to the fastq file.

    Yields:
        str: Read sequences.
    """
    with fastq_file.open('r', encoding='utf-8') as file:
        while True:
            try:
                next(file)  # Identifier line
                sequence = next(file).strip()  # Sequence line
                next(file)  # '+' line
                next(file)  # Quality line
                yield sequence
            except StopIteration:
                break


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    Args:
        read (str): Sequence of a read.
        kmer_size (int): Size of the kmers.

    Yields:
        str: Kmers of size kmer_size.
    """
    return (read[i:i + kmer_size] for i in range(len(read) - kmer_size + 1))


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionary object of all kmer occurrences in the fastq file.

    Args:
        fastq_file (Path): Path to the fastq file.
        kmer_size (int): Size of the kmers.

    Returns:
        Dict[str, int]: A dictionary object that identifies all kmer occurrences.
    """
    kmer_dict = {}
    for sequence in read_fastq(fastq_file):
        for kmer in cut_kmer(sequence, kmer_size):
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the De Bruijn graph.

    Args:
        kmer_dict (Dict[str, int]): A dictionary object that identifies all kmer occurrences.

    Returns:
        DiGraph: A directed graph (nx) of all kmer substrings and weights (occurrences).
    """
    graph = DiGraph()
    for kmer, weight in kmer_dict.items():
        prefix, suffix = kmer[:-1], kmer[1:]
        graph.add_edge(prefix, suffix, weight=weight)
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if not delete_entry_node:
            path = path[1:]
        if not delete_sink_node:
            path = path[:-1]

        # Remove the selected nodes
        graph.remove_nodes_from(path)

    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    weight_stdev = statistics.stdev(weight_avg_list)

    if weight_stdev > 0:
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    else:
        length_stdev = statistics.stdev(path_length)
        if length_stdev > 0:
            best_path_index = path_length.index(max(path_length))
        else:
            best_path_index = randint(0, len(path_list) - 1)

    del path_list[best_path_index]
    graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)

    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    paths = list(all_simple_paths(graph, ancestor_node, descendant_node))

    weight_list = [path_average_weight(graph, path) for path in paths]

    length_list = [len(path) for path in paths]

    graph = select_best_path(graph, paths, length_list, weight_list, delete_entry_node = False, delete_sink_node = False)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = True

    while bubble:
        bubble = False

        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))

            if len(predecessors) > 1:
                for i in range(len(predecessors)):
                    for j in range(i + 1, len(predecessors)):
                        ancestor = lowest_common_ancestor(graph, predecessors[i], predecessors[j])

                        if ancestor is not None:
                            bubble = True
                            graph = solve_bubble(graph, ancestor, node)
                            break

                    if bubble:
                        break
            if bubble:
                break
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    tip = True

    while tip:
        tip = False
        nodes = list(graph.nodes())  # Create a copy of the node list

        for node in nodes:
            if node not in graph:
                continue

            predecessors = list(graph.predecessors(node))
            entry_predecessors = []

            for pred in predecessors:
                for starting_node in starting_nodes:
                    if starting_node in graph and pred in graph:
                        if has_path(graph, starting_node, pred):
                            entry_predecessors.append(starting_node)
                            break

            if len(entry_predecessors) > 1:
                tip = True
                paths = []
                for pred in entry_predecessors:
                    if pred in graph and node in graph:
                        paths.extend(list(all_simple_paths(graph, pred, node)))

                if paths:  # Only proceed if we have valid paths
                    weight_list = [path_average_weight(graph, path) for path in paths]
                    length_list = [len(path) for path in paths]

                    graph = select_best_path(graph, paths, length_list, weight_list, delete_entry_node=True, delete_sink_node=False)

                break

        # Update starting_nodes list
        starting_nodes = [n for n in starting_nodes if n in graph]

    return graph

def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    tip = True

    while tip:
        tip = False
        nodes = list(graph.nodes())  # Create a copy of the node list

        for node in nodes:
            if node not in graph:
                continue

            successor = list(graph.successors(node))
            exit_successors = []

            for succ in successor:
                for ending_node in ending_nodes:
                    if ending_node in graph and succ in graph:
                        if has_path(graph, succ, ending_node):
                            exit_successors.append(ending_node)
                            break

            if len(exit_successors) > 1:
                tip = True
                paths = []
                for succ in exit_successors:
                    if succ in graph and node in graph:
                        paths.extend(list(all_simple_paths(graph, node, succ)))

                if paths:  # Only proceed if we have valid paths
                    weight_list = [path_average_weight(graph, path) for path in paths]
                    length_list = [len(path) for path in paths]

                    graph = select_best_path(graph, paths, length_list, weight_list, delete_entry_node=False, delete_sink_node=True)

                break

        # Update starting_nodes list
        ending_nodes = [n for n in ending_nodes if n in graph]

    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []

    for node in graph.nodes():
        if not list(graph.predecessors(node)):
            starting_nodes.append(node)

    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    ending_nodes = []

    for node in graph.nodes():
        if not list(graph.successors(node)):
            ending_nodes.append(node)

    return ending_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []

    for start in starting_nodes:
        for end in ending_nodes:
           if has_path(graph, start, end):
                # Find all simple paths between the start node and the end node
                for path in all_simple_paths(graph, start, end):
                    contig = path[0]

                    for node in path[1:]:
                        contig += node[-1]

                    contigs.append((contig, len(contig)))
            
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as f:
        for index, (contig, length) in enumerate(contigs_list):
            header = f">contig_{index} len={length}\n"
            f.write(header)
            
            formatted_sequence = textwrap.fill(contig, width=80)
            f.write(formatted_sequence + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Read the input file and build the graph
    print("Reading input file and building graph...")
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)

    # Simplify bubbles
    print("Simplifying bubbles...")
    graph = simplify_bubbles(graph)

    # Solve entry and exit tips
    print("Solving entry tips...")
    starting_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)

    print("Solving exit tips...")
    sink_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, sink_nodes)

    # Get contigs
    print("Generating contigs...")
    starting_nodes = get_starting_nodes(graph)
    sink_nodes = get_sink_nodes(graph)
    contigs = get_contigs(graph, starting_nodes, sink_nodes)

    # Save contigs
    print(f"Saving contigs to {args.output_file}...")
    save_contigs(contigs, args.output_file)

    print("Assembly complete!")
 
    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
