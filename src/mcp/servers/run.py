''''
Manage and run multiple MCP servers based on a configuration file.

Usage:
    python3 run.py

rowel.atienza@up.edu.ph
2025
'''

import os
import yaml
import time
from multiprocessing import Pool
import logging
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


#def run_server(name, transport, host, port, path, module, model=None, model_url=None):
def run_server(name:str, 
               transport:str, 
               host:str, 
               port:int, 
               path:str, 
               module:str, 
               options: dict={}) -> bool:
    """
    Start a server with the provided configuration.
    
    Args:
        name (str): Server name for identification
        transport (str): Transport protocol (e.g., 'sse')
        host (str): Host address to bind the server
        port (int): Port number to listen on
        path (str): URL path for the server endpoint
        module (str): Python module path to import and run
        model (str, optional): Model name if applicable
        model_url (str, optional): URL to download model if applicable
        
    Returns:
        bool: True if server started successfully, False otherwise
    """
    try:
        if 'stdio' in transport:
            logger.info(f"Starting {name} server using stdio transport")
        else:
            logger.info(f"Starting {name} server at {host}:{port} with path {path} using transport {transport}")
        
        if not module:
            logger.error(f"No module specified for server {name}")
            return False
            
        # Import the server module dynamically
        # Built-in shorthand: names starting with "tasks." or "src." are
        # resolved relative to the onit package.  Everything else is treated
        # as an absolute Python import path so that pip-installed third-party
        # packages work out of the box.
        if module.startswith("src."):
            full_module = module
        elif module.startswith("tasks."):
            full_module = f"src.mcp.servers.{module}"
        else:
            full_module = module  # third-party absolute module path
        server_module = __import__(full_module, fromlist=['run'])
        
        # Run the server
        server_module.run(
            transport=transport, 
            host=host, 
            port=port, 
            path=path, 
            options=options
        )
        
        logger.info(f"Server {name} started successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import module {module} for server {name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error starting {name} server: {e}")
        return False

def load_config(config_path=None, extra_configs=None):
    """
    Load server configuration from YAML file, optionally merging additional
    configs supplied via ``--mcp-config``.

    Args:
        config_path (str, optional): Path to the base configuration file.
            Defaults to the built-in ``configs/default.yaml``.
        extra_configs (list[str], optional): Paths to additional YAML config
            files whose ``servers`` lists are appended to the base config.
            This allows third parties to add their own MCP servers without
            modifying the default config.

    Returns:
        dict: Merged configuration with a ``servers`` list.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")

    logger.info(f"Loading configuration from {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

    # Merge additional third-party configs (--mcp-config)
    for extra_path in (extra_configs or []):
        logger.info(f"Merging extra MCP config from {extra_path}")
        if not os.path.exists(extra_path):
            raise FileNotFoundError(f"Extra config file not found at {extra_path}")
        try:
            with open(extra_path, 'r') as f:
                extra = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing extra YAML config {extra_path}: {e}")
            raise
        config.setdefault('servers', []).extend(extra.get('servers', []))

    return config

def prepare_server_args(config):
    """Extract server arguments from configuration."""
    server_args = []
    
    for server in config.get('servers', []):
        name = server.get('name')
        if not name:
            logger.warning("Skipping server with no name defined")
            continue
            
        transport = server.get('transport', 'sse')
        host = server.get('host', '0.0.0.0')
        port = server.get('port', 18201)
        path = server.get('path', f'/{name.lower()}')
        enabled = server.get('enabled', True)
        module = server.get('module')
        
        if not module:
            logger.warning(f"Skipping {name} server: No module specified")
            continue
       
        options = {}
        if 'options' in server:
            options = server.get('options', {})
            #model = kwargs.get('model')
            #model_url = kwargs.get('model_url')
            
        if enabled:
            server_args.append((name, transport, host, port, path, module, options))
            logger.info(f"Preparing to start {name} server at {host}:{port} with path {path}")
        else:
            logger.info(f"Skipping {name} server as it is disabled in the configuration.")
            
    return server_args

def run_servers(config_path=None, extra_configs=None, log_level='INFO'):
    """Run MCP servers based on a configuration file.

    Args:
        config_path (str, optional): Path to YAML config. Defaults to built-in default.
        extra_configs (list[str], optional): Additional config files to merge (--mcp-config).
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.getLogger().setLevel(getattr(logging, log_level))
    try:
        config = load_config(config_path, extra_configs=extra_configs)
        server_args = prepare_server_args(config)

        if not server_args:
            logger.warning("No enabled servers found in configuration")
            return

        with Pool(processes=len(server_args)) as pool:
            results = [pool.apply_async(run_server, args) for args in server_args]

            logger.info(f"Starting {len(server_args)} servers...")

            try:
                while True:
                    for i, result in enumerate(results):
                        if result.ready() and not result.get():
                            name = server_args[i][0]
                            logger.error(f"Server {name} failed to start or crashed")
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt, terminating all servers...")

    except Exception as e:
        logger.error(f"Failed to start servers: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MCP Server Manager")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()

    run_servers(config_path=args.config, log_level=args.log_level)
