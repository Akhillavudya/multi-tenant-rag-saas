"""Alembic environment.

Rewired from the generated default in two ways that matter:

1. The database URL comes from `app.config.settings`, NOT from alembic.ini.
   alembic.ini is committed, so a URL there would mean a password in git.
2. `target_metadata` points at our Base, so `alembic revision --autogenerate`
   can diff the models against the live schema.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings

# Importing the registry is what populates Base.metadata. Without this import
# autogenerate sees zero tables and cheerfully proposes dropping everything.
from app.db.models import Base

config = context.config

# Inject the real URL at runtime. set_main_option escapes '%' for us, which
# matters because URL-encoded passwords contain '%' and ConfigParser would
# otherwise treat it as interpolation syntax.
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    Useful when a DBA must review or apply the SQL by hand:
    `alembic upgrade head --sql`.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and run migrations against it."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: a migration run is a short-lived process. Pooling would
        # hold connections open for a script that is about to exit anyway.
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without compare_type, changing a column's type (VARCHAR(50) ->
            # VARCHAR(200)) is silently ignored by autogenerate.
            compare_type=True,
            # Likewise for server-side defaults.
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
