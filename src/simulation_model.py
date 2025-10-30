import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import minimize_scalar

def simulate_season_vectorized(
    begin_inventory: int,
    sale_price: float,
    unit_cost: float,
    salvage_value: float,
    people_visits: np.ndarray,
    thief_visits: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Vectorized season simulation with theft as a separate Poisson process.

    Parameters
    ----------
    begin_inventory : int
        Starting inventory Q for each simulation path.
    sale_price : float
        Selling price per unit.
    unit_cost : float
        Unit purchasing cost.
    salvage_value : float
        Salvage value per leftover unit at the end of the season.
    people_visits : np.ndarray  (num_simulations x num_days)
        Poisson(lambda) draws per day (customer arrival counts).
    thief_visits : np.ndarray    (num_simulations x num_days)
        Poisson(lambda * thief_pct) draws per day (theft counts).

    Returns
    -------
    neg_mean_profit : float
        Negative mean profit across simulations (useful for minimize()).
    mean_cust : float
        Mean total customer demand over the season.
    mean_theft : float
        Mean total theft over the season.
    std_cust : float
        Std of total customer demand over the season.
    std_theft : float
        Std of total theft over the season.
    """
    num_simulations, num_days = people_visits.shape

    # Inventory for each simulation path; track only current level.
    inventory = np.full((num_simulations,), begin_inventory, dtype=int)
    total_sales = np.zeros(num_simulations, dtype=float)

    # Iterate days (vectorized over simulations)
    for d in range(num_days):
        daily_demand = people_visits[:, d]    # customers wanting to buy
        daily_theft  = thief_visits[:, d]     # units stolen (attempted)

        # If unconstrained, all attempted theft and demand are satisfied from inventory if available;
        # If constrained (demand+theft exceeds inventory), allocate proportionally to observed mix.
        total_attempt = daily_demand + daily_theft

        unconstrained_mask = total_attempt <= inventory
        constrained_mask   = ~unconstrained_mask

        # Unconstrained case
        # All theft and demand attempts are realized; inventory reduced by both; revenue from demand only.
        theft_taken_uncon = np.where(unconstrained_mask, daily_theft, 0)
        sales_uncon       = np.where(unconstrained_mask, daily_demand, 0)
        inv_after_uncon   = inventory - (theft_taken_uncon + sales_uncon)

        # Constrained case: allocate inventory proportionally
        # theft_alloc = ceil(theft / (demand + theft) * inventory)
        # sales_alloc = inventory - theft_alloc
        theft_ratio = np.zeros_like(daily_theft, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            theft_ratio[constrained_mask] = np.divide(
                daily_theft[constrained_mask], 
                total_attempt[constrained_mask],
                out=np.zeros_like(daily_theft[constrained_mask], dtype=float),
                where=total_attempt[constrained_mask] > 0
            )

        theft_alloc = np.zeros_like(daily_theft, dtype=int)
        sales_alloc = np.zeros_like(daily_demand, dtype=int)
        theft_alloc[constrained_mask] = np.ceil(theft_ratio[constrained_mask] * inventory[constrained_mask]).astype(int)
        # prevent rounding from exceeding inventory
        theft_alloc = np.minimum(theft_alloc, inventory)
        sales_alloc[constrained_mask] = (inventory[constrained_mask] - theft_alloc[constrained_mask]).astype(int)

        # Combine unconstrained + constrained outcomes
        theft_taken = np.where(unconstrained_mask, theft_taken_uncon, theft_alloc)
        sales_made  = np.where(unconstrained_mask, sales_uncon, sales_alloc)

        # Update totals
        total_sales += sales_made
        inventory    = inventory - (theft_taken + sales_made)
        inventory    = np.maximum(inventory, 0)

    final_inventory = inventory

    # Profit per path: revenue from sales - purchase cost + salvage value on leftovers
    revenue = total_sales * sale_price
    purchase_cost = begin_inventory * unit_cost
    salvage_income = final_inventory * salvage_value
    profit = revenue - purchase_cost + salvage_income

    mean_profit = float(np.mean(profit))
    neg_mean_profit = -mean_profit  # for minimize()

    # Stats (across season)
    cust_totals = people_visits.sum(axis=1)
    theft_totals = thief_visits.sum(axis=1)
    mean_cust = float(np.mean(cust_totals))
    mean_theft = float(np.mean(theft_totals))
    std_cust = float(np.std(cust_totals))
    std_theft = float(np.std(theft_totals))

    return neg_mean_profit, mean_cust, mean_theft, std_cust, std_theft


def optimize_begin_inventory(
    lambda_value: float,
    num_simulations: int,
    season_length: int,
    thief_percentage: float,
    sale_price: float,
    cost_percentage: float,
    salvage_percentage: float,
    q_bounds: tuple[int, int] = (0, 2000),
    random_seed: int | None = None,
):
    """
    Build Poisson arrivals and theft arrays; optimize begin_inventory via scalar search.

    Returns
    -------
    result : dict
        Contains optimal Q, mean profit (positive), plus demand/theft stats.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        people_visits = rng.poisson(lam=lambda_value, size=(num_simulations, season_length))
        thief_visits  = rng.poisson(lam=lambda_value * thief_percentage, size=(num_simulations, season_length))
    else:
        people_visits = np.random.poisson(lambda_value, (num_simulations, season_length))
        thief_visits  = np.random.poisson(lambda_value * thief_percentage, (num_simulations, season_length))

    unit_cost = sale_price * cost_percentage
    salvage_value = unit_cost * salvage_percentage

    def objective(q):
        # minimize negative mean profit
        neg_mean_profit, *_ = simulate_season_vectorized(
            begin_inventory=int(q),
            sale_price=sale_price,
            unit_cost=unit_cost,
            salvage_value=salvage_value,
            people_visits=people_visits,
            thief_visits=thief_visits,
        )
        return neg_mean_profit

    res = minimize_scalar(
        objective,
        bounds=(q_bounds[0], q_bounds[1]),
        method="bounded"
    )

    q_star = int(round(res.x))
    neg_mean_profit, mc, mt, sc, st = simulate_season_vectorized(
        begin_inventory=q_star,
        sale_price=sale_price,
        unit_cost=unit_cost,
        salvage_value=salvage_value,
        people_visits=people_visits,
        thief_visits=thief_visits,
    )

    return {
        "season_length": season_length,
        "lambda_value": lambda_value,
        "thief_percentage": thief_percentage,
        "sale_price": sale_price,
        "cost_percentage": cost_percentage,
        "salvage_percentage": salvage_percentage,
        "Q_star": q_star,
        "mean_profit": -neg_mean_profit,  # convert back to positive
        "mean_customer_demand": mc,
        "mean_theft": mt,
        "std_customer_demand": sc,
        "std_theft": st,
    }


def grid_search_summary(
    lambda_value: float,
    num_simulations: int,
    season_lengths: list[int],
    thief_percentages: np.ndarray,
    sale_prices: list[float],
    cost_percentages: list[float],
    salvage_percentages: list[float],
    q_bounds: tuple[int, int] = (0, 2000),
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Loop across parameter grids and return a summary DataFrame of optimal Q and profit.
    """
    rows = []
    for season_length in season_lengths:
        for thief_pct in thief_percentages:
            for sp, cp, sv in product(sale_prices, cost_percentages, salvage_percentages):
                out = optimize_begin_inventory(
                    lambda_value=lambda_value,
                    num_simulations=num_simulations,
                    season_length=season_length,
                    thief_percentage=float(thief_pct),
                    sale_price=float(sp),
                    cost_percentage=float(cp),
                    salvage_percentage=float(sv),
                    q_bounds=q_bounds,
                    random_seed=random_seed,
                )
                rows.append(out)
    return pd.DataFrame(rows)
