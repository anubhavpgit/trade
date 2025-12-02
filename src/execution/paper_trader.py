"""
Paper trading execution engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

from loguru import logger

from src.config import settings
from src.data.models import (
    Order,
    Trade,
    OrderSide,
    OrderType,
    StockPosition,
)
from src.agent.hedging_agent import HedgeAction


@dataclass
class ExecutionResult:
    """Result of order execution."""

    success: bool
    trade: Optional[Trade]
    message: str
    slippage: float = 0.0
    commission: float = 0.0


@dataclass
class PaperAccount:
    """Paper trading account state."""

    cash: float = 100000.0
    stock_position: Optional[StockPosition] = None
    trades: list[Trade] = field(default_factory=list)
    pnl_realized: float = 0.0


class PaperTrader:
    """
    Paper trading execution engine.

    Features:
    - Simulated order execution
    - Slippage and spread modeling
    - Position tracking
    - Trade history
    """

    def __init__(self):
        self.config = settings.agent
        self.account = PaperAccount()
        self._current_price: float = 100.0

    def set_current_price(self, price: float):
        """Update current market price."""
        self._current_price = price
        if self.account.stock_position:
            self.account.stock_position.current_price = price

    async def execute(self, action: HedgeAction) -> ExecutionResult:
        """
        Execute a hedging action.

        Args:
            action: HedgeAction from the agent

        Returns:
            ExecutionResult with trade details
        """
        if not action.should_hedge:
            return ExecutionResult(
                success=True,
                trade=None,
                message="No hedge needed",
            )

        # Create order
        order = Order(
            symbol="SPY",  # Default underlying
            side=action.side,
            quantity=action.shares_to_trade,
            order_type=OrderType.MARKET,
        )

        return await self.execute_order(order)

    async def execute_order(self, order: Order) -> ExecutionResult:
        """
        Execute a single order.

        Args:
            order: Order to execute

        Returns:
            ExecutionResult with trade details
        """
        if not order.validate():
            return ExecutionResult(
                success=False,
                trade=None,
                message="Invalid order",
            )

        # Calculate execution price with slippage
        slippage_bps = 2.0  # 2 bps slippage
        if order.side == OrderSide.BUY:
            exec_price = self._current_price * (1 + slippage_bps / 10000)
        else:
            exec_price = self._current_price * (1 - slippage_bps / 10000)

        slippage = abs(exec_price - self._current_price) * order.quantity

        # Calculate commission
        commission = max(1.0, order.quantity * 0.005)  # $0.005/share, min $1

        # Create trade
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            timestamp=datetime.now(),
            commission=commission,
            slippage=slippage,
        )

        # Update account
        self._update_position(trade)
        self.account.trades.append(trade)

        logger.info(
            f"Executed: {trade.side.value} {trade.quantity} {trade.symbol} "
            f"@ ${trade.price:.2f} | Commission: ${commission:.2f}"
        )

        return ExecutionResult(
            success=True,
            trade=trade,
            message=f"Order executed successfully",
            slippage=slippage,
            commission=commission,
        )

    def _update_position(self, trade: Trade):
        """Update account position after trade."""
        signed_qty = (
            trade.quantity if trade.side == OrderSide.BUY else -trade.quantity
        )
        total_cost = trade.quantity * trade.price + trade.commission

        if trade.side == OrderSide.BUY:
            self.account.cash -= total_cost
        else:
            self.account.cash += trade.quantity * trade.price - trade.commission

        if self.account.stock_position is None:
            self.account.stock_position = StockPosition(
                symbol=trade.symbol,
                quantity=signed_qty,
                avg_cost=trade.price,
                current_price=self._current_price,
            )
        else:
            pos = self.account.stock_position
            old_qty = pos.quantity
            new_qty = old_qty + signed_qty

            if new_qty == 0:
                # Position closed
                realized_pnl = (
                    trade.price - pos.avg_cost
                ) * trade.quantity * (1 if trade.side == OrderSide.SELL else -1)
                self.account.pnl_realized += realized_pnl
                self.account.stock_position = None
            elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                # Adding to position
                pos.avg_cost = (
                    pos.avg_cost * abs(old_qty) + trade.price * trade.quantity
                ) / abs(new_qty)
                pos.quantity = new_qty
            else:
                # Flipping position
                realized_pnl = (
                    trade.price - pos.avg_cost
                ) * abs(old_qty) * (1 if trade.side == OrderSide.SELL else -1)
                self.account.pnl_realized += realized_pnl
                pos.quantity = new_qty
                pos.avg_cost = trade.price

    async def close_all_positions(self):
        """Close all positions (kill switch)."""
        logger.warning("Closing all positions...")

        if self.account.stock_position:
            pos = self.account.stock_position
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            order = Order(
                symbol=pos.symbol,
                side=side,
                quantity=abs(pos.quantity),
                order_type=OrderType.MARKET,
            )
            await self.execute_order(order)

    def get_position(self) -> Optional[StockPosition]:
        """Get current stock position."""
        return self.account.stock_position

    def get_account_summary(self) -> dict:
        """Get account summary."""
        pos = self.account.stock_position
        position_value = pos.market_value if pos else 0
        unrealized_pnl = pos.pnl if pos else 0

        return {
            "cash": self.account.cash,
            "position_value": position_value,
            "total_value": self.account.cash + position_value,
            "realized_pnl": self.account.pnl_realized,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": self.account.pnl_realized + unrealized_pnl,
            "num_trades": len(self.account.trades),
        }

    def get_trade_history(self) -> list[dict]:
        """Get trade history."""
        return [
            {
                "trade_id": t.trade_id,
                "timestamp": t.timestamp.isoformat(),
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "commission": t.commission,
            }
            for t in self.account.trades
        ]
