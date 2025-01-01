# Import backtester tests
from tests.test_backtester import (
    test_backtester_initialization,
    test_backtester_with_strategy,
    test_backtester_with_multiple_strategies,
    test_backtester_rebalancing_daily,
    test_backtester_rebalancing_intraday,
    test_backtester_with_costs,
    test_backtester_invalid_frequency,
    test_backtester_frequency_validation,
)

# Import result tests
from tests.test_result import (
    test_result_initialization,
    test_invalid_positions,
    test_calculate_nav,
    test_get_metrics_per_asset,
    test_plotting_multi_asset,
    test_compare_results_multi_asset,
    test_different_asset_comparison,
)

# Import strategy tests
from tests.test_strategy import (
    test_strategy_decorator,
    test_ma_crossover,
    test_rsi_strategy,
    test_strategy_position_bounds,
    test_strategy_parameters,
    test_strategy_with_missing_data,
    test_custom_strategy,
)

# Import data utility tests
from tests.test_data_utils import (
    test_data_loading_from_file,
    test_data_validation,
    test_data_frequency_detection,
    test_data_format_standardization,
    test_multi_asset_data_alignment,
)

# Import metrics tests
from tests.test_metrics import (
    test_annualized_return,
    test_annualized_std,
    test_sharpe_ratio,
    test_sortino_ratio,
    test_max_drawdown,
    test_count_trades,
    test_winning_trades,
)
