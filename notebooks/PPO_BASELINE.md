# PPO Baseline — Opis Implementacji

## Co to jest?

Dodaliśmy baseline oparty na **PPO (Proximal Policy Optimization)** z biblioteki Stable-Baselines3.
Jest to punkt 5 z planu Karola: *"Porównanie z Deep RL"*.

Agent PPO trenuje się na losowych labiryntach, a następnie używa wyuczonej polityki
do eksploracji — zero-shot, na nowym labiryncie. Dzięki temu możemy porównać go
z naszymi heurystykami opartymi o SIREN (MaxVar, Frontier).

## Instalacja zależności

Przed uruchomieniem zainstaluj:

```bash
pip install stable-baselines3 gymnasium
```

## Pliki

| Plik | Opis |
|------|------|
| `src/agent/strategy/ppo.py` | Implementacja strategii PPO + wrapper środowiska Gymnasium |
| `configs/agent/rl_ppo.yaml` | Konfiguracja Hydra dla agenta PPO |

## Jak uruchomić?

### Podstawowe uruchomienie (200k kroków treningowych)
```bash
python main.py agent=rl_ppo
```

### Z własną liczbą kroków treningowych
```bash
python main.py agent=rl_ppo agent.strategy.total_timesteps=1000000
```

### Zapisanie modelu pod własną nazwą (zalecane!)
```bash
python main.py agent=rl_ppo agent.strategy.total_timesteps=1000000 agent.strategy.model_path=outputs/ppo_1M
```

### Dwa eksperymenty po sobie (Hydra multirun)
```bash
python main.py -m agent=rl_ppo agent.strategy.total_timesteps=200000,1000000 agent.strategy.model_path=outputs/ppo_200k,outputs/ppo_1M
```

### Combo z innymi opcjami (np. unconstrained)
```bash
python main.py agent=rl_ppo metric=downstream
```

## Jak to działa?

### 1. Faza treningu (przed eksploracją)
PPO trenuje się na **losowo generowanych labiryntach** (tych samych co reszta projektu,
generowanych przez `mazelib`). Agent uczy się polityki eksploracji: dostaje nagrodę za
odkrywanie nowych komórek i karę za uderzanie w ściany.

Obserwacja agenta RL to:
- Spłaszczona mapa pamięci (`-1` = nieznane, `0` = wolne, `1` = ściana)
- Pozycja agenta (znormalizowana do `[-1, 1]`)

Akcja to: indeks komórki docelowej (`y * size + x`).

### 2. Faza eksploracji (zero-shot)
Po treningu wytrenowana polityka PPO jest używana jako strategia w miejsce MaxVar/Frontier.
Przy każdym wywołaniu `select_action()` polityka dostaje aktualny stan mapy i zwraca cel.

**Fallback:** Jeśli PPO wybierze komórkę nieosiągalną (poza zasięgiem ruchu),
strategia automatycznie przełącza się na greedy max-uncertainty dla tego kroku.

### 3. Cache modelu
Model jest zapisywany do pliku `.zip` po treningu. Przy kolejnym uruchomieniu
z tą samą ścieżką (`model_path`) jest wczytywany — trening nie powtarza się.

Żeby wymusić nowy trening: usuń plik `.zip` lub zmień `model_path`.

## Wyniki (PoC, labirynt 65×65, 10 kroków)

| Strategia | PSNR (krok 10) | Uwagi |
|-----------|---------------|-------|
| MaxVar (baseline) | ~4.61 dB | — |
| PPO 200k kroków | ~4.25 dB | model trenował ~9 min |
| PPO 1M kroków | TBD | model trenował ~40 min |

**Interpretacja:** PPO przy ograniczonym budżecie kroków (10 makro-kroków) nie
pokonuje heurystyk opartych o Active Learning. To zgodne z literaturą — RL potrzebuje
wielu interakcji żeby nauczyć się długoterminowego planowania. Jest to wartościowy
wynik do raportu: pokazuje że nasze podejście SIREN + MaxVar jest kompetytywne
względem end-to-end RL przy małym budżecie eksploracji.

## Parametry konfiguracji (`configs/agent/rl_ppo.yaml`)

```yaml
strategy:
  total_timesteps: 200000   # liczba kroków treningowych PPO
  n_envs: 4                 # równoległe środowiska treningowe
  model_path: outputs/ppo_policy  # ścieżka zapisu/odczytu modelu
```
