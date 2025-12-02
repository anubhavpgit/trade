"""
Order management and routing.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from loguru import logger

from src.data.models import Order, Trade, OrderSide, OrderType


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class ManagedOrder:
    """Order with management metadata."""

    order_id: str
    order: Order
    status: OrderStatus
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    rejection_reason: Optional[str] = None


class OrderManager:
    """
    Order management system.

    Features:
    - Order validation
    - Order state tracking
    - Cancellation handling
    - Fill reporting
    """

    def __init__(self):
        self._orders: dict[str, ManagedOrder] = {}
        self._pending_orders: list[str] = []

    def create_order(self, order: Order) -> ManagedOrder:
        """
        Create a managed order.

        Args:
            order: Order specification

        Returns:
            ManagedOrder with tracking
        """
        order_id = str(uuid.uuid4())[:12]

        managed = ManagedOrder(
            order_id=order_id,
            order=order,
            status=OrderStatus.PENDING,
        )

        self._orders[order_id] = managed
        self._pending_orders.append(order_id)

        logger.debug(f"Created order {order_id}: {order.side.value} {order.quantity}")
        return managed

    def submit_order(self, order_id: str) -> bool:
        """
        Mark order as submitted.

        Returns:
            True if successfully submitted
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        logger.debug(f"Submitted order {order_id}")
        return True

    def fill_order(
        self,
        order_id: str,
        filled_quantity: int,
        fill_price: float,
    ) -> bool:
        """
        Record order fill.

        Args:
            order_id: Order ID
            filled_quantity: Quantity filled
            fill_price: Fill price

        Returns:
            True if fill recorded
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
            return False

        # Update fill
        total_filled = order.filled_quantity + filled_quantity
        order.avg_fill_price = (
            order.avg_fill_price * order.filled_quantity
            + fill_price * filled_quantity
        ) / total_filled
        order.filled_quantity = total_filled

        if order.filled_quantity >= order.order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            if order_id in self._pending_orders:
                self._pending_orders.remove(order_id)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        logger.debug(
            f"Filled order {order_id}: {filled_quantity} @ ${fill_price:.2f}"
        )
        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Returns:
            True if successfully cancelled
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            return False

        order.status = OrderStatus.CANCELLED
        if order_id in self._pending_orders:
            self._pending_orders.remove(order_id)

        logger.debug(f"Cancelled order {order_id}")
        return True

    def reject_order(self, order_id: str, reason: str) -> bool:
        """
        Reject an order.

        Returns:
            True if rejection recorded
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason

        if order_id in self._pending_orders:
            self._pending_orders.remove(order_id)

        logger.warning(f"Rejected order {order_id}: {reason}")
        return True

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_pending_orders(self) -> list[ManagedOrder]:
        """Get all pending/submitted orders."""
        return [self._orders[oid] for oid in self._pending_orders]

    def get_order_history(self, limit: int = 100) -> list[ManagedOrder]:
        """Get recent order history."""
        orders = list(self._orders.values())
        orders.sort(
            key=lambda o: o.submitted_at or datetime.min,
            reverse=True,
        )
        return orders[:limit]

    def cancel_all_pending(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        count = 0
        for order_id in self._pending_orders.copy():
            if self.cancel_order(order_id):
                count += 1
        return count
