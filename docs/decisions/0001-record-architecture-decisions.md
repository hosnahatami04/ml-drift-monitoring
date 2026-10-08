# 0001. Record architecture decisions

Date: 2026-10-08

## Status

Accepted

## Context

This project makes a number of non-obvious choices: hand-written statistical
tests instead of off-the-shelf drift libraries, a shared monitoring core used
by both the live service and the evaluation harness, strict seed separation
between calibration and evaluation. Six months from now the reasoning behind
these choices will be gone unless it is written down next to the code.

## Decision

Every non-trivial decision gets a short architecture decision record in
`docs/decisions/`, numbered `NNNN-title.md`, with four sections: context,
decision, alternatives considered, consequences. One page maximum. Records are
immutable once merged; a reversal is a new record that supersedes the old one.

## Alternatives considered

- Design notes in the README: gets stale, mixes audience (users vs.
  maintainers), and loses the one-decision-one-record granularity.
- Full RFC process: too heavy for a single-maintainer project.

## Consequences

Pull requests that introduce a significant choice must link the matching
record. Reviewing history becomes cheaper; writing the record costs a few
minutes per decision.
