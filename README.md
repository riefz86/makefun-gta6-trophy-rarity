# makefun-gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/riefz86/makefun-gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/riefz86/makefun-gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PS5 trophy's Exophase earn rate a week after launch, as five quantiles on the logit scale. A trophy's rate is earners divided by players tracked, and the two move for different reasons — new players dilute every rate while the players already tracked keep earning — so the program forecasts them separately and divides: log players and log earners are each linear in log(1 + hours since launch), fitted by weighted least squares on the case's reads, with each trophy's slope shrunk toward its tier's median slope. The spread is the honest predictive uncertainty on the logit scale: fit variance, plus a jump term for Exophase's refreshes, plus model risk growing with the horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9hts85as&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
