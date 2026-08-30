PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('reader', 'note', 'tablet')),
    release_year INTEGER,
    screen_size_inches REAL NOT NULL,
    is_color INTEGER NOT NULL DEFAULT 0 CHECK (is_color IN (0, 1)),
    mono_ppi INTEGER,
    color_ppi INTEGER,
    operating_system TEXT,
    is_open_system INTEGER NOT NULL DEFAULT 0 CHECK (is_open_system IN (0, 1)),
    supports_stylus INTEGER NOT NULL DEFAULT 0 CHECK (supports_stylus IN (0, 1)),
    has_front_light INTEGER NOT NULL DEFAULT 0 CHECK (has_front_light IN (0, 1)),
    has_warm_light INTEGER NOT NULL DEFAULT 0 CHECK (has_warm_light IN (0, 1)),
    is_waterproof INTEGER NOT NULL DEFAULT 0 CHECK (is_waterproof IN (0, 1)),
    storage_gb INTEGER,
    weight_g INTEGER,
    notes TEXT,
    spec_source_url TEXT,
    spec_verified_at TEXT,
    UNIQUE (brand, model)
);

CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER NOT NULL,
    price_cny INTEGER NOT NULL CHECK (price_cny >= 0),
    seller TEXT NOT NULL,
    product_url TEXT,
    collected_at TEXT NOT NULL,
    is_available INTEGER NOT NULL DEFAULT 1 CHECK (is_available IN (0, 1)),
    FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_devices_screen_size
ON devices(screen_size_inches);

CREATE INDEX IF NOT EXISTS idx_devices_color_stylus
ON devices(is_color, supports_stylus);

CREATE INDEX IF NOT EXISTS idx_offers_device_price
ON offers(device_id, price_cny);
