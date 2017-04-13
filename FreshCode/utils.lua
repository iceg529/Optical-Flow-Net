-- utility functions from nvidia digits tools digits/tools/torch

require 'torch'   -- torch
require 'image'   -- for color transforms

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

-- round function
function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

-- return whether a Luarocks module is available
function isModuleAvailable(name)
  if package.loaded[name] then
    return true
  else
    for _, searcher in ipairs(package.searchers or package.loaders) do
      local loader = searcher(name)
      if type(loader) == 'function' then
        package.preload[name] = loader
        return true
      end
    end
    return false
  end
end

-- attempt to require a module and drop error if not available
function check_require(name)
    if isModuleAvailable(name) then
        return require(name)
    else
        assert(false,"Did you forget to install " .. name .. " module? c.f. install instructions")
    end
end
