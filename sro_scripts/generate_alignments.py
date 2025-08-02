#!/usr/bin/env python3
"""
generate_alignments.py

This script reads FASTA files (one sequence per file) from a specified directory,
runs the alignment generation (using HHSearch/hmmsearch and jackhmmer/hhblits),
and writes the results to an alignment directory. This script is intended to precompute
alignments for later inference.
"""

import argparse
import os
import logging
import random
import numpy as np
import time

from openfold.data import templates, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.utils.script_utils import parse_fasta

from scripts.utils import add_data_args
from log_utils import configure_logging
logger = configure_logging()

def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")
        local_alignment_dir = os.path.join(alignment_dir, tag)
        logger.info(f"Generating alignments for {tag}...")
        os.makedirs(local_alignment_dir, exist_ok=True)

        # Select the appropriate template searcher based on config preset
        if "multimer" in args.config_preset:
            template_searcher = hmmsearch.Hmmsearch(
                binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                database_path=args.pdb_seqres_database_path,
            )
        else:
            template_searcher = hhsearch.HHSearch(
                binary_path=args.hhsearch_binary_path,
                databases=[args.pdb70_database_path],
            )

        # Set up the alignment runner based on whether single sequence mode is requested
        if args.use_single_seq_mode:
            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                template_searcher=template_searcher,
                no_cpus=args.cpus,
            )
        else:
            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniref30_database_path=args.uniref30_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                uniprot_database_path=args.uniprot_database_path,
                template_searcher=template_searcher,
                use_small_bfd=args.bfd_database_path is None,
                no_cpus=args.cpus,
            )

        # Run the alignment generation
        alignment_runner.run(tmp_fasta_path, local_alignment_dir)
        os.remove(tmp_fasta_path)


def list_files_with_extensions(directory, extensions):
    return [f for f in os.listdir(directory) if f.endswith(extensions)]


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    alignment_dir = args.output_dir

    # Process each FASTA file in the input directory
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        logger.info(f"Processing FASTA file: {fasta_path}")
        with open(fasta_path, "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        # If a file contains more than one sequence and you're not in multimer mode,
        # you might want to skip or process only the first entry.
        if not args.use_single_seq_mode and len(tags) != 1:
            logger.warning(f"{fasta_path} contains more than one sequence. Only the first will be processed.")
            tags = tags[:1]
            seqs = seqs[:1]

        precompute_alignments(tags, seqs, alignment_dir, args)
        logger.info(f"Alignments generated for sequences in {fasta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate precomputed alignments for FASTA sequences."
    )
    parser.add_argument("fasta_dir", type=str, help="Directory containing FASTA files, one sequence per file.")
    parser.add_argument("template_mmcif_dir", type=str, help="Directory containing template mmCIF files.")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(),
                        help="Directory to output alignments.")
    parser.add_argument("--config_preset", type=str, default="model_1",
                        help="Model configuration preset (e.g. 'model_1' or 'model_3').")
    parser.add_argument("--use_single_seq_mode", action="store_true", default=False,
                        help="Use single sequence embeddings instead of MSAs.")
    parser.add_argument("--use_precomputed_alignments", type=str, default=None,
                        help="If provided, alignment computation is skipped and alignments are loaded from this path.")
    parser.add_argument("--cpus", type=int, default=4,
                        help="Number of CPUs to use for alignment tools.")
    add_data_args(parser)

    args = parser.parse_args()

    main(args)
