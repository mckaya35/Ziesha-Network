"""İşlem kuralı ve performans metrikleri."""
import numpy as np
import pandas as pd

C, LAM, HOURS = 0.001, 2.0, 8760


def positions(pred, c=C, lam=LAM):
    """Bar kapanışında sinyal; |tahmin| > λc ise pozisyon değişir, aksi hâlde korunur (long-only)."""
    pos, cur = np.empty(len(pred)), 0.0
    for i, p in enumerate(pred):
        if abs(p) > lam * c:
            cur = 1.0 if p > 0 else 0.0
        pos[i] = cur
    return pos


def net_returns(pos, y, fund, c=C):
    """pos[t] bar t kapanışında alınır, y[t] (sonraki bar) getirisini kazanır. Funding long iken düşülür."""
    pos = np.asarray(pos, float)
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pos * y - c * turn - pos * fund, int((turn > 0).sum())


def funding_per_bar(index, funding):
    """y[t] aralığı t+1h..t+2h; o aralığın sonunda (t+2h) ödenen funding'i eşle."""
    f = funding["rate"].copy()
    f.index = f.index.round("h")
    f = f.groupby(level=0).sum()
    return f.reindex(index + pd.Timedelta(hours=2)).fillna(0.0).to_numpy()


def metrics(r, trades, fold_ids):
    r = pd.Series(r)
    eq = np.exp(r.cumsum())
    folds = r.groupby(np.asarray(fold_ids)).sum()
    return {
        "Yıllık getiri": np.exp(r.mean() * HOURS) - 1,
        "Yıllık vol": r.std() * np.sqrt(HOURS),
        "Sharpe": r.mean() / r.std() * np.sqrt(HOURS) if r.std() > 0 else 0.0,
        "Maks. düşüş": (eq / eq.cummax() - 1).min(),
        "İşlem": trades,
        "Kârlı fold": f"{int((folds > 0).sum())}/{len(folds)}",
    }


def random_positions(n_bars, n_trades, rng):
    """Nakitten başlar, rastgele n_trades noktada pozisyonu çevirir."""
    flips = np.zeros(n_bars)
    flips[rng.choice(n_bars, size=min(n_trades, n_bars), replace=False)] = 1
    return np.cumsum(flips) % 2
