# FAI Final Project: Texas Hold'em Poker Agent

Individual final project for **CSIE 5043 / Foundations of Artificial Intelligence (Spring 2025)** at National Taiwan University.

This repository contains the Texas Hold'em poker agents developed on top of the course's PyPokerEngine-based game environment, along with the project specification ([final-proj.pdf](final-proj.pdf)) and final report ([report.pdf](report.pdf)).

See [report.pdf](report.pdf) for the agent designs (preflop table lookup, Monte Carlo variants, D3QN) and full evaluation results against the course baselines. The final submitted agent is [`agents/monteoc_player.py`](agents/monteoc_player.py) (Monte Carlo with over-commit detection).

## Running

The compiled course baselines (`baseline0`–`baseline7`) target **Python 3.8 on Linux x86_64**.

```bash
pip install -r requirement.txt
python3 start_game.py       # single match (configure players in start_game.py)
python3 judge.py            # evaluate against baselines
```

## Notes

- The game engine (`game/`), compiled baseline binaries (`baseline*.so`), and assignment spec belong to the course staff / PyPokerEngine authors (see [LICENSE.md](LICENSE.md)).
