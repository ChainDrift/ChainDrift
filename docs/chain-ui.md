# ChainUI

Chaindrift provides a builtin webserver, which can serve [ChainUI](https://github.com/chaindrift/chainui), the chaindrift frontend.

By default, the UI is automatically installed as part of the installation (script, docker).
chainUI can also be manually installed by using the `chaindrift install-ui` command.
This same command can also be used to update chainUI to new new releases.

Once the bot is started in trade / dry-run mode (with `chaindrift trade`) - the UI will be available under the configured API port (by default `http://127.0.0.1:8080`).

??? Note "Looking to contribute to chainUI?"
    Developers should not use this method, but instead clone the corresponding use the method described in the [chainUI repository](https://github.com/chaindrift/chainui) to get the source-code of chainUI. A working installation of node will be required to build the frontend.

!!! tip "chainUI is not required to run chaindrift"
    chainUI is an optional component of chaindrift, and is not required to run the bot.
    It is a frontend that can be used to monitor the bot and to interact with it - but chaindrift itself will work perfectly fine without it.

## Configuration

ChainUI does not have it's own configuration file - but assumes a working setup for the [rest-api](rest-api.md) is available.
Please refer to the corresponding documentation page to get setup with chainUI

## UI

ChainUI is a modern, responsive web application that can be used to monitor and interact with your bot.

ChainUI provides a light, as well as a dark theme.
Themes can be easily switched via a prominent button at the top of the page.
The theme of the screenshots on this page will adapt to the selected documentation Theme, so to see the dark (or light) version, please switch the theme of the Documentation.

### Login

The below screenshot shows the login screen of chainUI.

![ChainUI - login](assets/chainui-login-CORS.png#only-dark)
![ChainUI - login](assets/chainui-login-CORS-light.png#only-light)

!!! Hint "CORS"
    The Cors error shown in this screenshot is due to the fact that the UI is running on a different port than the API, and [CORS](#cors) has not been setup correctly yet.

### Trade view

The trade view allows you to visualize the trades that the bot is making and to interact with the bot.
On this page, you can also interact with the bot by starting and stopping it and - if configured - force trade entries and exits.

![ChainUI - trade view](assets/chainUI-trade-pane-dark.png#only-dark)
![ChainUI - trade view](assets/chainUI-trade-pane-light.png#only-light)

### Plot Configurator

ChainUI Plots can be configured either via a `plot_config` configuration object in the strategy (which can be loaded via "from strategy" button) or via the UI.
Multiple plot configurations can be created and switched at will - allowing for flexible, different views into your charts.

The plot configuration can be accessed via the "Plot Configurator" (Cog icon) button in the top right corner of the trade view.

![ChainUI - plot configuration](assets/chainUI-plot-configurator-dark.png#only-dark)
![ChainUI - plot configuration](assets/chainUI-plot-configurator-light.png#only-light)

### Settings

Several UI related settings can be changed by accessing the settings page.

Things you can change (among others):

* Timezone of the UI
* Visualization of open trades as part of the favicon (browser tab)
* Candle colors (up/down -> red/green)
* Enable / disable in-app notification types

![ChainUI - Settings view](assets/chainui-settings-dark.png#only-dark)
![ChainUI - Settings view](assets/chainui-settings-light.png#only-light)

## Backtesting

When chaindrift is started in [webserver mode](utils.md#webserver-mode) (chaindrift started with `chaindrift webserver`), the backtesting view becomes available.
This view allows you to backtest strategies and visualize the results.

You can also load and visualize previous backtest results, as well as compare the results with each other.

![ChainUI - Backtesting](assets/chainUI-backtesting-dark.png#only-dark)
![ChainUI - Backtesting](assets/chainUI-backtesting-light.png#only-light)


--8<-- "includes/cors.md"
