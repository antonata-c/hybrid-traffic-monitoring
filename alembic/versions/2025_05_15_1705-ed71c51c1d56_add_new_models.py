"""add new models

Revision ID: ed71c51c1d56
Revises: 9e58cc19faa5
Create Date: 2025-05-15 17:05:09.908213

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ed71c51c1d56'
down_revision: Union[str, None] = '9e58cc19faa5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('network_links',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('source_node', sa.String(), nullable=False),
    sa.Column('target_node', sa.String(), nullable=False),
    sa.Column('bandwidth', sa.Float(), nullable=False),
    sa.Column('latency', sa.Float(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('link_type', sa.String(), nullable=False),
    sa.Column('weight', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_network_links_source_node'), 'network_links', ['source_node'], unique=False)
    op.create_index(op.f('ix_network_links_target_node'), 'network_links', ['target_node'], unique=False)
    op.create_table('nodes',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('node_id', sa.String(), nullable=False),
    sa.Column('node_type', sa.Enum('fiber', 'gen5', 'satellite', 'microwave', 'starlink', 'hybrid', name='nodetype'), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('location', sa.String(), nullable=True),
    sa.Column('coordinates', sa.JSON(), nullable=True),
    sa.Column('max_capacity', sa.Float(), nullable=False),
    sa.Column('backup_node', sa.String(), nullable=True),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_nodes_node_id'), 'nodes', ['node_id'], unique=True)
    op.create_index(op.f('ix_nodes_node_type'), 'nodes', ['node_type'], unique=False)
    op.create_table('optimization_actions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('action_type', sa.String(), nullable=False),
    sa.Column('affected_nodes', sa.JSON(), nullable=False),
    sa.Column('description', sa.String(), nullable=False),
    sa.Column('before_metrics', sa.JSON(), nullable=False),
    sa.Column('after_metrics', sa.JSON(), nullable=True),
    sa.Column('improvement', sa.Float(), nullable=True),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_optimization_actions_action_type'), 'optimization_actions', ['action_type'], unique=False)
    op.create_table('qos_policies',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('node_id', sa.String(), nullable=False),
    sa.Column('traffic_type', sa.Enum('VOICE', 'VIDEO', 'INTERACTIVE', 'STREAMING', 'DATA', 'IOT', 'SIGNALING', name='traffictype'), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('bandwidth_reserved', sa.Float(), nullable=True),
    sa.Column('max_latency', sa.Float(), nullable=True),
    sa.Column('max_packet_loss', sa.Float(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_qos_policies_node_id'), 'qos_policies', ['node_id'], unique=False)
    op.create_index(op.f('ix_qos_policies_traffic_type'), 'qos_policies', ['traffic_type'], unique=False)
    op.add_column('traffic', sa.Column('jitter', sa.Float(), nullable=False))
    op.add_column('traffic', sa.Column('signal_strength', sa.Float(), nullable=True))
    op.add_column('traffic', sa.Column('interference_level', sa.Float(), nullable=True))
    op.add_column('traffic', sa.Column('error_rate', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('traffic', 'error_rate')
    op.drop_column('traffic', 'interference_level')
    op.drop_column('traffic', 'signal_strength')
    op.drop_column('traffic', 'jitter')
    op.drop_index(op.f('ix_qos_policies_traffic_type'), table_name='qos_policies')
    op.drop_index(op.f('ix_qos_policies_node_id'), table_name='qos_policies')
    op.drop_table('qos_policies')
    op.drop_index(op.f('ix_optimization_actions_action_type'), table_name='optimization_actions')
    op.drop_table('optimization_actions')
    op.drop_index(op.f('ix_nodes_node_type'), table_name='nodes')
    op.drop_index(op.f('ix_nodes_node_id'), table_name='nodes')
    op.drop_table('nodes')
    op.drop_index(op.f('ix_network_links_target_node'), table_name='network_links')
    op.drop_index(op.f('ix_network_links_source_node'), table_name='network_links')
    op.drop_table('network_links')
    # ### end Alembic commands ###
